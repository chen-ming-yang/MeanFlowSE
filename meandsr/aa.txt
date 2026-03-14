Training

dys_ssl.npy (B, T_sys, 1024) ---> PerceiverBottleneck --> bottleneck (B, 64, 512)
pre-extracted, variable len       variable -> fixed
                                  produce fixed len output

use length predictor from bottleneck to pred_length

(z_x) norm_vae.npy (B, T_norm, 256) ---> MeanFlow sampling
pre-extracted, variable len         z_t = (1 - t) * noise + t * z_x
                                        |
                                        |
                                    DiTBlockWithCrossAttention
                                        self_attn(z_t)
                                        cross_attn(z_t, bottleneck)
                                        feedforward
                                    
                                    get u_hat 
                                    u_target = z_normal - noise
                                    cal masked_adaptive_l2_loss


Inference
dysarthric_wav --> SSLEncoder --> PerceiverBottleneck --> bottleneck
                                                            |
                                                        lengthPredictor -> T_normal
                                                            |
                                noise ~ N(0, I) at T_normal --> DiT(noise, bottleneck, r=0, t=1)
                                                            |
                                                    z_0 = noise - u_hat
                                                            |
                                                    VAEDecoder(z_0) --> normal_wav