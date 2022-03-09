import torch
import torch.nn as nn

from faceformer.embedding import PositionEmbeddingLearned, VanillaEmedding
from faceformer.transformer import (TransformerDecoder,
                                    TransformerDecoderLayer,
                                    TransformerEncoder,
                                    TransformerEncoderLayer)
from faceformer.utils import min_value_of_dtype


class SurfaceFormer_Parallel(nn.Module):

    def __init__(self, num_model=512, num_head=8, num_feedforward=2048, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 dropout=0.1, activation="relu", normalize_before=True,
                 num_points_per_line=50, num_lines=64, point_dim=2, 
                 max_face_length=10, token=None, 
                 teacher_forcing_ratio=0, **kwargs):
        super(SurfaceFormer_Parallel, self).__init__()

        self.num_model = num_model # E
        self.max_face_length = max_face_length # T
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.token = token
        self.num_token = token.len

        self.val_enc = VanillaEmedding(num_points_per_line * point_dim, num_model, token)

        # position encoding
        self.pos_enc = PositionEmbeddingLearned(num_model, max_len=num_lines+self.num_token)
        self.query_pos_enc = PositionEmbeddingLearned(num_model, max_len=max_face_length)

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
        for _, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def get_embeddings(self, input, label):
        val_embed = self.val_enc(input)
        pos_embed = self.pos_enc(val_embed)
        query_pos_embed = self.query_pos_enc(label.transpose(1, 2))

        return val_embed, pos_embed, query_pos_embed
    
    def process_masks(self, input_mask, tgt_mask=None):
        # pad input mask
        padding_mask = torch.zeros((len(input_mask), self.num_token)).type_as(input_mask)
        input_mask = torch.cat([padding_mask, input_mask], dim=1)
        if tgt_mask is None:
            return input_mask
        # tgt is 1 shorter
        tgt_mask = tgt_mask[..., :-1].contiguous()
        return input_mask, tgt_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (1 - torch.tril(torch.ones(sz, sz))) == 1
        return mask

    def patch_source(self, src, pos):
        src = src.transpose(0, 1)
        pos = pos.transpose(0, 1)
        return src, pos

    def patch_target(self, tgt, pos):
        tgt = tgt.permute(2, 0, 1)
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
        # target: N x F x T
        input, input_mask = inputs['input'], inputs['input_mask']
        label, label_mask = inputs['label'], inputs['label_mask']
        max_num_edges = max(inputs['num_input'])
        label, label_mask = label[:, :max_num_edges, :], label_mask[:, :max_num_edges, :]

        # process masks
        input_mask, label_mask = self.process_masks(input_mask, label_mask)

        # embeddings: N x L x E, L+=4
        val_embed, pos_embed, query_pos_embed = self.get_embeddings(input, label)

        # prepare data: L x N x E
        source, pos_embed = self.patch_source(val_embed, pos_embed)
        # target: T x N x (F)
        target, label, query_pos_embed = self.patch_target(label, query_pos_embed)

        # encoder: L x N x E
        memory = self.encoder(source, src_key_padding_mask=input_mask, pos=pos_embed)

        # L x NF x E
        memory = memory.repeat_interleave(max_num_edges, 1)

        # feature gather: T x N x E
        tgt_mask = self.generate_square_subsequent_mask(target.size(0)).type_as(input_mask)

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
        
        # T x NF x E
        target = target.unsqueeze(-1).repeat(1, 1, 1, self.num_model).flatten(1, 2)
        # memory: L x NF x E
        # target: T x NF x E
        # tgt: T x NF x E
        tgt = torch.gather(memory, 0, target) # selects the targeting edge features
        # NF x T
        label_mask = label_mask.flatten(0, 1)
        # NF x L
        input_mask = input_mask.repeat_interleave(max_num_edges, 0)

        # decoder: T x NF x E
        pointer = self.decoder(tgt, memory, tgt_mask=tgt_mask, pos=pos_embed, query_pos=query_pos_embed,
                               tgt_key_padding_mask=label_mask, memory_key_padding_mask=input_mask)

        pointer = self.project(pointer)

        # outputs
        inputs['embedding'] = memory.transpose(0, 1)
        inputs['pointer'] = pointer.transpose(0, 1)
        inputs['label'] = label.flatten(1, 2).transpose(0, 1)
        return inputs

    def select_next(self, embedding, pointer, input_mask):
        embedding = embedding.transpose(0, 1)
        pointer = pointer.permute(1, 2, 0)
        logit = torch.bmm(embedding, pointer[..., -1:])
        logit = logit.masked_fill(input_mask.unsqueeze(-1), min_value_of_dtype(logit.dtype))
        next_token = torch.argmax(logit, dim=1).transpose(0, 1)
        return next_token

    def forward_eval(self, inputs):
        # inputs: N x L x P x D
        input, input_mask = inputs['input'], inputs['input_mask']
        label = inputs['label']

        batch_size = input.size(0) # N
        max_num_edges = max(inputs['num_input']) # L

        
        # process masks
        input_mask = self.process_masks(input_mask)

        # vertex embedding: N x L x E, L+=4
        val_embed, pos_embed, query_pos_embed = self.get_embeddings(input, label)

        # prepare data: L x N x E
        source, pos_embed = self.patch_source(val_embed, pos_embed)
        
        # use all edges as the first token
        # anchors: 1 x N x (F)
        anchors = torch.arange(max_num_edges).repeat(1, batch_size, 1).type_as(label)

        # mask unused face seq start token as EOS (Other type of face)
        for i, num_edges in enumerate(inputs['num_input']):
            anchors[:, i, num_edges:] = self.token.len - 1
        query_pos_embed = query_pos_embed.transpose(0, 1)
        predicts = anchors.flatten(1, 2) # 1 x N(F)

        # encoder: L x N x E
        memory = self.encoder(source, src_key_padding_mask=input_mask, pos=pos_embed)
        # L x N(F) x E
        memory = memory.repeat_interleave(max_num_edges, 1)
        # N(F) x L
        input_mask = input_mask.repeat_interleave(max_num_edges, 0)

        for step in range(self.max_face_length - 1):
            target = predicts.unsqueeze(-1).repeat(1, 1, self.num_model)

            tgt = torch.gather(memory, 0, target)

            # decoder: T x N(F) x E
            pointer = self.decoder(tgt, memory, memory_key_padding_mask=input_mask,
                                   pos=pos_embed, query_pos=query_pos_embed[:step+1])

            pointer = self.project(pointer)

            next_token = self.select_next(memory, pointer, input_mask)

            predicts = torch.cat((predicts, next_token), dim=0)
            
            # if all face tokens are EOS or PAD, stop decoding
            if torch.all(next_token < self.num_token):
                break
        
        # pad predict to self.max_face_length
        predicts = torch.cat((predicts, torch.zeros(self.max_face_length - predicts.size(0), predicts.size(1)).type_as(predicts)), dim=0)

        #predicts: T x N(F)

        inputs['predict'] = predicts.transpose(0, 1).view(-1, max_num_edges, self.max_face_length)
        return inputs

    def forward(self, inputs):
        """
            inputs:
                input       (N x L x P x D),
                label       (N x F x T),
                input_mask  (N x L),
                label_mask  (N x F x T)
            outputs:
                embedding   (NF x L x E),
                pointer     (NF x T x E),
                label       (N x F x T)
        """
        if self.training:
            outputs = self.forward_train(inputs)
        else:
            outputs = self.forward_eval(inputs)
        return outputs
