import tensorflow as tf
from tcn import TCN

def tcn_model_26(timesteps, input_dim, output_dim):
    i = tf.keras.Input(batch_shape=(None, timesteps, input_dim))
    o = TCN(
        nb_filters=256,
        kernel_size=2,
        nb_stacks=2,
        dilations=[1, 2, 4, 8, 16, 32],
        padding="causal",
        use_skip_connections=True,
        dropout_rate=0.0,
        return_sequences=False,
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=True,
    )(i)
    o = tf.keras.layers.Dense(output_dim)(o)
    model = tf.keras.models.Model(inputs=[i], outputs=[o])
    return model


model = tcn_model_26(
        params["max_length"], len(params["features"]), len(params["targets"])
    )
model.compile(optimizer=params["optimizer"], loss=loss)