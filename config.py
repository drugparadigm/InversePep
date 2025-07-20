import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # Misc Config
    config.exp_type = 'vpsde'
    config.device   = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.seed     = 42
    config.save     = True

    # Data config
    config.data                = data = ml_collections.ConfigDict()
    data.seq_centered          = True
    data.radius                = 4.5
    data.top_k                 = 10
    data.num_rbf               = 16
    data.num_posenc            = 16
    data.num_conformers        = 1
    data.add_noise             = -1.0
    data.knn_num               = 10
    
    # SDE / noise schedule
    config.sde                  = sde = ml_collections.ConfigDict()
    sde.schedule               = 'cosine'     # 'linear' or 'cosine'
    sde.continuous_beta_0      = 0.1
    sde.continuous_beta_1      = 20.0

    # Sampling
    config.sampling             = sampling = ml_collections.ConfigDict()
    sampling.method            = 'ancestral'
    sampling.steps             = 200        

    # Model (GVPTransCond)
    config.model               = model = ml_collections.ConfigDict()
    model.geometric_data_parallel = False
    model.ema_decay            = 0.999
    model.pred_data            = True
    model.self_cond            = True
    model.name                 = 'GVPTransCond'

    # Node input dims: (scalar_channels, vector_channels)
    model.node_in_dim          = (54, 2)
    model.node_h_dim           = (512, 128)
    model.edge_in_dim          = (36, 1) 
    model.edge_h_dim           = (128, 1)
    model.num_layers           = 4
    model.drop_rate            = 0.2

    # (20 standard + X unknown + PAD slot) 
    model.out_dim              = 22

    # Time and dihedral conditioning, etc.
    model.time_cond            = True
    model.dihedral_angle       = False
    model.num_trans_layer      = 8
    model.drop_struct          = -1.

    # Transformer sub‐config
    model.trans                = trans = ml_collections.ConfigDict()
    trans.encoder_embed_dim     = 512
    trans.encoder_attention_heads = 16
    trans.attention_dropout     = 0.2
    trans.dropout               = 0.2
    trans.encoder_ffn_embed_dim = 1024

    # Optimization
    config.optim               = optim = ml_collections.ConfigDict()
    optim.weight_decay         = 0.01
    optim.optimizer            = 'AdamW'
    optim.lr                   = 1e-4
    optim.beta1                = 0.9
    optim.eps                  = 1e-8
    optim.warmup               = 20000
    optim.grad_clip            = 20.0
    optim.disable_grad_log     = True

    # Evaluation
    config.eval                = ev = ml_collections.ConfigDict()
    ev.model_path              = ''
    ev.test_perplexity         = False
    ev.test_recovery           = True
    ev.n_samples               = 10
    ev.sampling_steps          = 200
    ev.cond_scale              = 3.0
    ev.temperature             = 0.7
    ev.dynamic_threshold       = True
    ev.dynamic_thresholding_percentile = 0.90

    # Training
    config.train               = train = ml_collections.ConfigDict()
    train.batch_size           = 16        
    train.epochs               = 100
    train.num_workers          = 4
    train.self_cond_prob       = 0.5        # p(self-conditioning)
    train.drop_struct_prob     = 0.5        # p(drop structure)
    train.log_every            = 100
    train.save_every           = 10         
    train.ckpt_dir             = './Enhanced_ckpts_latest_3'
    model.drop_struct          = train.drop_struct_prob

    optim.lr_min = 1e-6        
    optim.lr_decay_epochs = train.epochs  


    return config