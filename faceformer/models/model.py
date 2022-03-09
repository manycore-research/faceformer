import torch
import torch.nn as nn

from faceformer.embedding import PositionEmbeddingLearned, VanillaEmedding
from faceformer.transformer import (TransformerDecoder,
                                    TransformerDecoderLayer,
                                    TransformerEncoder,
                                    TransformerEncoderLayer)
from faceformer.utils import min_value_of_dtype


class SurfaceFormer(nn.Module):

    def __init__(self, num_model=512, num_head=8, num_feedforward=2048, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 dropout=0.1, activation="relu", normalize_before=True,
                 num_points_per_line=50, num_lines=1000, point_dim=2, 
                 label_seq_length=2000, token=None, teacher_forcing_ratio=0, **kwargs):
        super(SurfaceFormer, self).__init__()

        self.num_model = num_model # E
        self.num_labels = label_seq_length # T
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.token = token
        self.num_token = token.len

        self.val_enc = VanillaEmedding(num_points_per_line * point_dim, num_model, token)

        # position encoding
        self.pos_enc = PositionEmbeddingLearned(num_model, max_len=num_lines+self.num_token)
        self.query_pos_enc = PositionEmbeddingLearned(num_model, max_len=label_seq_length)

        # vertex transformer encoder
        encoder_layers = TransformerEncoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(num_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        # wire transformer decoder
        decoder_layers = TransformerDecoderLayer(
            num_model, num_head, num_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(num_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm)

        self.project = nn.Linear(num_model, num_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def get_embeddings(self, input, label):
        val_embed = self.val_enc(input)
        pos_embed = self.pos_enc(val_embed)
        query_pos_embed = self.query_pos_enc(label)

        return val_embed, pos_embed, query_pos_embed
    
    def process_masks(self, input_mask, tgt_mask=None):
        # pad input mask
        padding_mask = torch.zeros((len(input_mask), self.num_token), device=input_mask.device).type_as(input_mask)
        input_mask = torch.cat([padding_mask, input_mask], dim=1)
        if tgt_mask is None:
            return input_mask
        # tgt is 1 shorter
        tgt_mask = tgt_mask[:, :-1].contiguous()
        return input_mask, tgt_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (1 - torch.tril(torch.ones(sz, sz))) == 1
        return mask

    def patch_source(self, src, pos):
        src = src.transpose(0, 1)
        pos = pos.transpose(0, 1)
        return src, pos

    def patch_target(self, tgt, pos):
        tgt = tgt.transpose(0, 1)
        tgt, label = tgt[:-1].contiguous(), tgt[1:].contiguous()
        pos = pos.transpose(0, 1)
        pos = pos[:-1].contiguous()
        return tgt, label, pos

    def mix_gold_sampled(self, gold_target, sampled_target, prob):
        sampled_target = torch.cat((gold_target[0:1], sampled_target[:-1]), dim=0)

        targets = torch.stack((gold_target, sampled_target))

        random = torch.rand(gold_target.shape, device=targets.device)
        index = (random < prob).long().unsqueeze(0)

        new_target = torch.gather(targets, 0, index)
        return new_target.squeeze(0)

    def forward_train(self, inputs, scheduled_sampling_ratio=0):
        # inputs: N x L x P x D
        input, input_mask = inputs['input'], inputs['input_mask']
        label, label_mask = inputs['label'], inputs['label_mask']

        # process masks
        input_mask, label_mask = self.process_masks(input_mask, label_mask)

        # embeddings: N x L x E, L+=4
        val_embed, pos_embed, query_pos_embed = self.get_embeddings(input, label)

        # prepare data: L x N x E
        source, pos_embed = self.patch_source(val_embed, pos_embed)
        target, label, query_pos_embed = self.patch_target(label, query_pos_embed)

        # encoder: L x N x E
        memory = self.encoder(source, src_key_padding_mask=input_mask, pos=pos_embed)

        # feature gather: T x N x E
        tgt_mask = self.generate_square_subsequent_mask(target.size(0)).to(target.device)

        # T x N
        gold_target = target

        if scheduled_sampling_ratio > 0:
            with torch.no_grad():
                target = target.unsqueeze(-1).repeat(1, 1, self.num_model)

                tgt = torch.gather(memory, 0, target)

                # decoder: T x N x E
                pointer = self.decoder(tgt, memory, tgt_mask=tgt_mask, pos=pos_embed, query_pos=query_pos_embed,
                                       tgt_key_padding_mask=label_mask, memory_key_padding_mask=input_mask)

                pointer = self.project(pointer)

                logits = torch.bmm(memory.transpose(0, 1), pointer.permute(1, 2, 0))

                logits = logits.masked_fill(input_mask.unsqueeze(-1), min_value_of_dtype(logits.dtype))

                sampled_target = torch.argmax(logits, dim=1).transpose(0, 1)

                target = self.mix_gold_sampled(gold_target, sampled_target, scheduled_sampling_ratio)


        target = target.unsqueeze(-1).repeat(1, 1, self.num_model)
        # memory: L x N x E
        # target: T x N x E
        # tgt: T x N x E
        tgt = torch.gather(memory, 0, target) # selects the targeting edge features

        # decoder: T x N x E
        pointer = self.decoder(tgt, memory, tgt_mask=tgt_mask, pos=pos_embed, query_pos=query_pos_embed,
                               tgt_key_padding_mask=label_mask, memory_key_padding_mask=input_mask)

        pointer = self.project(pointer)

        # outputs
        inputs['embedding'] = memory.transpose(0, 1)
        inputs['pointer'] = pointer.transpose(0, 1)
        inputs['label'] = label.transpose(0, 1)
        return inputs

    def select_next(self, embedding, pointer, input_mask):
        embedding = embedding.transpose(0, 1)
        pointer = pointer.permute(1, 2, 0)
        logit = torch.bmm(embedding, pointer[..., -1:])
        logit = logit.masked_fill(input_mask.unsqueeze(-1), min_value_of_dtype(logit.dtype))
        next_token = torch.argmax(logit, dim=1).transpose(0, 1)
        return next_token

    def forward_eval(self, inputs):
        # inputs
        input, input_mask = inputs['input'], inputs['input_mask']
        label = inputs['label']

        batch_size = input.size(0)
        
        # process masks
        input_mask = self.process_masks(input_mask)

        # vertex embedding: N x L x E, L+=4
        val_embed, pos_embed, query_pos_embed = self.get_embeddings(input, label)

        # prepare data: L x N x E
        source, pos_embed = self.patch_source(val_embed, pos_embed)
        query_pos_embed = query_pos_embed.transpose(0, 1)

        # encoder: L x N x E
        memory = self.encoder(source, src_key_padding_mask=input_mask, pos=pos_embed)

        # T x N
        predicts = torch.full((1, batch_size), self.token.SOS, dtype=torch.long).to(source.device)
        EOS_found = 0

        for step in range(self.num_labels - 1):
            target = predicts.unsqueeze(-1).repeat(1, 1, self.num_model)

            tgt = torch.gather(memory, 0, target)

            # decoder: T x N x E
            pointer = self.decoder(tgt, memory, memory_key_padding_mask=input_mask,
                                   pos=pos_embed, query_pos=query_pos_embed[:step+1])

            pointer = self.project(pointer)

            next_token = self.select_next(memory, pointer, input_mask)

            predicts = torch.cat((predicts, next_token), dim=0)
            EOS_found += next_token.eq(self.token.EOS).sum().item()

            if EOS_found == batch_size:
                break
        
        # pad predict to self.num_labels
        predicts = torch.cat((predicts, torch.zeros(self.num_labels - predicts.size(0), predicts.size(1)).type_as(predicts)), dim=0)

        
        inputs['embedding'] = memory.transpose(0, 1)
        inputs['pointer'] = pointer.transpose(0, 1)
        inputs['predict'] = predicts.transpose(0, 1)
        return inputs

    def forward(self, inputs):
        """
            inputs:
                input       (N x L x P x D),
                label       (N x T),
                input_mask  (N x L),
                label_mask  (N x T)
            outputs:
                embedding   (N x L x E),
                pointer     (N x T x E),
                label       (N x T)
        """
        if self.training:
            outputs = self.forward_train(inputs)
        else:
            outputs = self.forward_eval(inputs)
        return outputs
