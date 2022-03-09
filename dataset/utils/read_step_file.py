import os

import timeout_decorator
from OCC.Core.IFSelect import IFSelect_ItemsByEntity, IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import list_of_shapes_to_compound


@timeout_decorator.timeout(5, use_signals=False)
def read_step_file(filename, as_compound=True, verbosity=True, filter_num_shape=10):
    """ read the STEP file and returns a compound and number of shapes
    filename: the file path
    verbosity: optional, False by default.
    as_compound: True by default. If there are more than one shape at root,
    gather all shapes into one compound. Otherwise returns a list of shapes.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError("%s not found." % filename)

    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == IFSelect_RetDone:  # check status
        if verbosity:
            failsonly = False
            step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
            step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
        transfer_result = step_reader.TransferRoots()
        if not transfer_result:
            raise AssertionError("Transfer failed.")
        _nbs = step_reader.NbShapes()
        if _nbs == 0:
            raise AssertionError("No shape to transfer.")
        elif _nbs == 1:  # most cases
            return step_reader.Shape(1), _nbs
        elif _nbs > 1:
            if _nbs > filter_num_shape:
                return None, _nbs
            shps = []
            # loop over root shapes
            for k in range(1, _nbs + 1):
                new_shp = step_reader.Shape(k)
                if not new_shp.IsNull():
                    shps.append(new_shp)
            if as_compound:
                compound, result = list_of_shapes_to_compound(shps)
                if not result:
                    print("Warning: all shapes were not added to the compound")
                return compound, _nbs
            else:
                print("Warning, returns a list of shapes.")
                return shps, _nbs
    else:
        raise AssertionError("Error: can't read file.")
    return None
