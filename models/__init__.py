def create_model(opt):
    model = None

    if opt.model == 'DANN_m':
        #assert(opt.dataset_mode == 'aligned')
        from .DANN_m_model import DANN_m_Model
        model = DANN_m_Model()

    elif opt.model == 'DSN_m':
        #assert(opt.dataset_mode == 'aligned')
        from .DSN_m_model import DSN_m_Model
        model = DSN_m_Model()

    elif opt.model == 'DSN_mv2':
        #assert(opt.dataset_mode == 'aligned')
        from .DSN_m_modelv2 import DSN_m_Modelv2
        model = DSN_m_Modelv2()

    elif opt.model == 'DANN_mv2':
        #assert(opt.dataset_mode == 'aligned')
        from .DANN_m_modelv2 import DANN_m_Modelv2
        model = DANN_m_Modelv2()

    elif opt.model == 'Di_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_m import Di_Model
        model = Di_Model()

    elif opt.model == 'Di_DSN_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_DSN_m import Di_DSN_Model
        model = Di_DSN_Model()

    elif opt.model == 'Di_mv2':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_mv2 import Di_Modelv2
        model = Di_Modelv2()

    elif opt.model == 'Di_mv3':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_mv3 import Di_Modelv3
        model = Di_Modelv3()

    elif opt.model == 'DANN_m_iter':
        # assert(opt.dataset_mode == 'aligned')
        from .DANN_m_iter_model import DANN_m_iter_Model
        model = DANN_m_iter_Model()

    elif opt.model == 'DSN_m_iter':
        # assert(opt.dataset_mode == 'aligned')
        from .DSN_m_iter_model import DSN_m_iter_Model
        model = DSN_m_iter_Model()

    elif opt.model == 'Di_iter_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_iter_m import Di_iter_Model
        model = Di_iter_Model()

    elif opt.model == 'Di_iter_DSN_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_iter_DSN_m import Di_iter_DSN_Model
        model = Di_iter_DSN_Model()

    elif opt.model == 'CORAL_m':
        # assert(opt.dataset_mode == 'aligned')
        from .CORAL_m_model import CORAL_m_Model
        model = CORAL_m_Model()

    elif opt.model == 'Di_CORAL_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_CORAL_m import Di_CORAL_Model
        model = Di_CORAL_Model()

    elif opt.model == 'CORAL_m_iter':
        # assert(opt.dataset_mode == 'aligned')
        from .CORAL_m_iter_model import CORAL_m_iter_Model
        model = CORAL_m_iter_Model()

    elif opt.model == 'Di_iter_CORAL_m':
        #assert(opt.dataset_mode == 'aligned')
        from .Di_iter_CORAL_m import Di_iter_CORAL_Model
        model = Di_iter_CORAL_Model()

    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
