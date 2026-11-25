"""Type-embedding inputs of a public-family instance (option --pub_type_source).

A factory instance carries an operation type and a station type per node, and
the released factory recipe feeds both to the policy through the type
embedding (model/main_model.py, use_type_embedding). A public instance has
neither, so the public arms run with --use_type_embedding False and observe no
attribute of a job.

--pub_type_source selects what the type embedding receives on a public instance:

  none   (default) no type input; the public arms as trained so far. Every code
         path that calls type_lists() with 'none' passes None, exactly as before.
  class  the operation-type input of an operation is the setup class of its job,
         0 .. N_CLASSES - 1, read from the descriptor (ConstraintDescriptor.op_class);
         the station-type input is one constant type, 0, on every machine. The
         network is built with use_type_embedding True and embedding tables of
         CLASS_N_OP_TYPES and CLASS_N_MCH_TYPES rows.

Every public descriptor carries op_class in every combination, also in N, L, W
and LW, where no setup is charged: the class is drawn from its own seed stream
(constraint_family/public_rules.py, draw_rules) and shared by all combinations
of a base instance. In those combinations the class input is a label without a
physical effect.

numpy only; no params import.
"""
import numpy as np

from constraint_family.descriptor import N_CLASSES

PUB_TYPE_SOURCES = ('none', 'class')
CLASS_N_OP_TYPES = N_CLASSES       # one embedding row per setup class
CLASS_N_MCH_TYPES = 1              # one constant station type
# Parameter count of the public recipe (10x5+pubfamily, the configuration of the
# pubx arms) with --pub_type_source class: the 29,602 of the class-blind network,
# plus the two embedding tables (4 x 8 + 1 x 8 = 40), plus 3,072 in the first
# attention layers, which read 8 more input channels on operations and machines.
# (The factory tables, 5 operation and 9 station types, give 32,786.)
PUBC_PARAMS = 32714
PUBX_PARAMS = 29602


def check_source(source):
    if source not in PUB_TYPE_SOURCES:
        raise ValueError(f'pub_type_source must be one of {PUB_TYPE_SOURCES}, '
                         f'got {source!r}')
    return source


def type_lists(source, descriptors, n_m):
    """(op_type_list, mch_type_list) for FJSPEnvFamily.set_initial_data.

    'none' -> (None, None). 'class' -> one int array [N] per descriptor holding
    the job class of each operation, and one all-zero int array [n_m] per
    descriptor."""
    check_source(source)
    if source == 'none':
        return None, None
    ops, mchs = [], []
    for d in descriptors:
        c = np.asarray(d.op_class, dtype=int)
        if c.ndim != 1 or c.size == 0:
            raise ValueError('a descriptor without op_class cannot feed the class input')
        if c.min() < 0 or c.max() >= CLASS_N_OP_TYPES:
            raise ValueError(f'op_class outside 0..{CLASS_N_OP_TYPES - 1}: '
                             f'{c.min()}..{c.max()}')
        ops.append(c.copy())
        mchs.append(np.zeros(int(n_m), dtype=int))
    return ops, mchs


def snapshot_source(snap):
    """The type source a training configuration snapshot was trained with;
    snapshots written before the option existed carry no key and mean 'none'."""
    v = snap.get('pub_type_source', None)
    return 'none' if v in (None, '', 'None') else check_source(v)


def refuse_mismatch(model_source, option_source, model_name=''):
    """SystemExit unless a checkpoint's type source equals the evaluation option."""
    if model_source != option_source:
        raise SystemExit(
            f'REFUSING: {model_name or "the policy"} was trained with '
            f'--pub_type_source {model_source}, the evaluation asks for '
            f'--pub_type_source {option_source}. Pass --pub_type_source '
            f'{model_source} to evaluate it.')
