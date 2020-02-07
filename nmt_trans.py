import tensorflow as tf
import numpy as np
import pandas as pd


class self_attn(tf.keras.Model):
    
    def __init__(self, q, k, v, mask_flag):
        super(self_attn, self).__init__()
        self.WQ = tf.keras.layers.Dense(q)
        self.WK = tf.keras.layers.Dense(k)
        self.WV = tf.keras.layers.Dense(v)
        self.mask_flag = mask_flag
    
    def call(self, x, mask):
        
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        
        self_atten_out = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, perm  = [0,2,1]))/np.sqrt(tf.shape(K)[2]))
        
        if self.mask_flag:
            self_atten_out = self_atten_out*mask    
        
        W_t_self_atten = tf.matmul(self_atten_out,V)
        
        return W_t_self_atten

class multi_head_atten(tf.keras.Model):
    
    def __init__(self, q, k, v, h, mask_flag):
        super(multi_head_atten, self).__init__()
        self.multilayers = [self_attn(q, k, v, mask_flag) for i in range(0,h)]
        self.linear_layer = tf.keras.layers.Dense(h*v)
        
    def call(self, x, mask):
        
        self_attention_outputs = []
        
        for layer in self.multilayers:
            self_attention_outputs.append(layer(x, mask))
        
        multi_output = tf.concat(axis = 2, values = self_attention_outputs)
        
        return self.linear_layer(multi_output)

class EncoderLayer(tf.keras.Model):
    
    def __init__(self, q, k, v, h):
        super(EncoderLayer, self).__init__()
        self.multi_head_layer = multi_head_atten(q, k, v, h, False)
        self.ff_layer = tf.keras.layers.Dense(h*v)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=2)
    
    def call(self, x, mask):
            
        atten = self.multi_head_layer(x, None)
        out = self.layernorm(x + atten)
        out_ff = self.ff_layer(out)
        out_encod = self.layernorm(out + out_ff)
        
        mask = tf.cast(mask, dtype=out_encod.dtype)
        out_encod = out_encod*mask  
        
        return out_encod

class Encoder(tf.keras.Model):
    
    def __init__(self, src_vocab_size, q, k, v, h, d_model, encoder_stacks):
        super(Encoder, self).__init__() 
        self.enocderlayers = [EncoderLayer(q, k, v, h) for stcks in range(0,encoder_stacks)]
        self.embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
    
    def call(self, x, pos_encoding):
        
        mask = tf.math.logical_not(tf.math.equal(x, 0))
        mask = tf.expand_dims(mask, axis = 2)
        
        x = self.embedding(x)
        
        x = x + pos_encoding
        
        encoder_outputs = []
        layer_input = x
        
        for layer in self.enocderlayers:
            
            encoder_outputs.append(layer(layer_input, mask))
            layer_input = encoder_outputs[-1]
        
        return encoder_outputs

class DecoderLayer(tf.keras.Model):
    
     def __init__(self, q, k, v, h):
        super(DecoderLayer, self).__init__()
        self.multi_head_layer_masked = multi_head_atten(q, k, v , h, True)
        self.multi_head_layer_non_masked = multi_head_atten(q, k, v , h, False)
        self.ff_layer = tf.keras.layers.Dense(h*v)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=2)
    
     def call(self, x, mask, encoder_out, layer_no):
         
        # intem 1
        atten = self.multi_head_layer_masked(x, mask)
        out = self.layernorm(x + atten)
        out_layer_1a = tf.concat([encoder_out, out], axis = 2)
        
        # intem 2
        atten = self.multi_head_layer_non_masked(out_layer_1a, None)
        out = self.layernorm(out + atten)
        
        out_ff = self.ff_layer(out)
        
        out_decod = self.layernorm(out + out_ff)

        return out_decod   
    
class Deocoder(tf.keras.Model):
    
    def __init__(self, tgt_vocab_size, q, k, v, h, d_model, decoder_stacks):
        super(Deocoder, self).__init__() 
        self.decoderlayers = [DecoderLayer(q, k, v, h) for stcks in range(0,decoder_stacks)]
        self.embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

    def call(self, x, pos_encoding, encoder_outs):
        
        max_pad = tf.shape(x)[1]
        
        mask = np.tril(np.ones((max_pad,max_pad)))
        
        x = self.embedding(x)
        
        x = x + pos_encoding
        
        decoder_outputs = []
        layer_input = x
        
        for layer_no, layer in enumerate(self.decoderlayers):
            
            decoder_outputs.append(layer(layer_input, mask, encoder_outs[layer_no], layer_no))
            layer_input = decoder_outputs[-1]
        
        return decoder_outputs
    
class Transformer(tf.keras.Model):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, q, k, v, h, d_model, encoder_stacks, decoder_stacks):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(src_vocab_size, q, k, v, h, d_model, encoder_stacks)
        self.Decoder = Deocoder(tgt_vocab_size, q, k, v, h, d_model, decoder_stacks)
        self.final_layer = tf.keras.layers.Dense(tgt_vocab_size)
    
    def call(self, source, target, pos_encoding):
        
        encoder_out = self.Encoder(source, pos_encoding)
        decoder_out = self.Decoder(source, pos_encoding, encoder_out)
        
        trans_out  = self.final_layer(decoder_out[-1])
        return trans_out

    @staticmethod
    def positional_encoding(d_model, max_pad, batch_size):
        
        sin_cos = 1/np.power(10000,(2*(np.arange(d_model)[np.newaxis, :]//2))/d_model)
    
        pos_encoding = np.arange(max_pad)[:, np.newaxis]
        
        pos_encoding = pos_encoding*sin_cos
        
        pos_encoding[:, 0::2] = np.cos(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2]= np.sin(pos_encoding[:, 1::2])
        
        return pos_encoding[np.newaxis, :]
    
    


src_vocab_size, tgt_vocab_size, q, k, v, h, d_model, encoder_stacks, decoder_stacks = 50000, 50000, 5, 5, 5, 2, 10, 2, 2

Trans = Transformer(src_vocab_size, tgt_vocab_size, q, k, v, h, d_model, encoder_stacks, decoder_stacks)

pos.shape = tf.cast(Trans.positional_encoding(d_model, 59, 2), tf.float32)

o = tf.nn.softmax(Trans(src_pad1, tgt_pad1, pos))



