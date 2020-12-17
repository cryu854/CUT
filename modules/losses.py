""" Implement the following loss functions that used in CUT/FastCUT model.
GANLoss
PatchNCELoss
"""

import tensorflow as tf


class GANLoss:
    def __init__(self, gan_mode):
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = tf.keras.losses.MeanSquaredError()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented.')
        
    def __call__(self, prediction, target_is_real):

        if self.gan_mode == 'lsgan':
            if target_is_real:
                loss = self.loss(prediction, tf.ones_like(prediction))
            else:
                loss = self.loss(prediction, tf.zeros_like(prediction))

        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = tf.reduce_mean(tf.math.softplus(-prediction))
            else:
                loss = tf.reduce_mean(tf.math.softplus(prediction))
                
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = tf.reduce_mean(-prediction)
            else:
                loss = tf.reduce_mean(prediction)
        return loss


class PatchNCELoss:
    def __init__(self, nce_temp, nce_lambda):
        # Potential: only supports for batch_size=1 now.
        self.nce_temp = nce_temp
        self.nce_lambda = nce_lambda
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
                                        reduction=tf.keras.losses.Reduction.NONE,
                                        from_logits=True)

    def __call__(self, source, target, netE, netF):
        feat_source = netE(source, training=True)
        feat_target = netE(target, training=True)

        feat_source_pool, sample_ids = netF(feat_source, training=True)
        feat_target_pool, _ = netF(feat_target, patch_ids=sample_ids, training=True)

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(feat_source_pool, feat_target_pool):
            n_patches, dim = feat_s.shape
            
            # pos logit
            l_pos = tf.matmul(tf.reshape(feat_s, (n_patches, 1, -1)), tf.reshape(feat_t, (n_patches, -1, 1)))
            l_pos = tf.squeeze(l_pos, 1)

            # neg logit
            l_neg = tf.matmul(feat_s, tf.transpose(feat_t))
 
            # diagonal entries are similarity between same features, and hence meaningless.
            # just fill the diagonal with very small number, which is exp(-10) and almost zero
            diagonal = tf.eye(n_patches, dtype=tf.bool)
            l_neg = tf.where(diagonal, tf.math.exp(-10.0), l_neg)

            out = tf.concat([l_pos, l_neg], axis=1) / self.nce_temp
            target = tf.concat([tf.ones_like(l_pos), tf.zeros_like(l_neg)], axis=1)

            loss = self.cross_entropy_loss(target, out) * self.nce_lambda
            total_nce_loss += tf.reduce_mean(loss)

        return total_nce_loss / len(feat_source_pool)
