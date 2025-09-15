import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput

from torch.nn import Linear, LayerNorm
from transformers.models.bart.modeling_bart import BartAttention

 
from convnext.convnext import convnext_tiny
 
import pandas as pd 

 

class_map_path = "./audioset_classes_embeddings/class_labels_indices.csv"
class_df = pd.read_csv(class_map_path)
index2mid = dict(zip(class_df["index"], class_df["mid"]))
index2name = dict(zip(class_df["index"], class_df["display_name"]))
 

 
#######################  Keywords(Optional) ######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_audio_enc = convnext_tiny(
    pretrained=False,
    strict=False,
    drop_path_rate=0.0,
    after_stem_dim=[252, 56],
    use_speed_perturb=False
)

state_dict = torch.load(
    "./convnext/convnext_tiny_471mAP.pth",
    map_location=device
)

pretrained_audio_enc.load_state_dict(state_dict['model'])
pretrained_audio_enc.to(device)    
for p in pretrained_audio_enc.parameters():
    p.requires_grad = False

pretrained_audio_enc.eval()          
#############################################################################################



class BARTAAC(nn.Module):
    def __init__(self, settings, device):
        super().__init__()
        
        self.device = device
        

        state_dict = torch.load(settings['lm']['audio_enc_path'], map_location=device)
       
        self.audio_enc = convnext_tiny(pretrained=False, strict=False, drop_path_rate=0.0, after_stem_dim=[252, 56], use_speed_perturb=False)
        state_dict = torch.load(settings['lm']['audio_enc_path'], map_location=device)
        self.audio_enc.load_state_dict(state_dict['model'])
        self.freeze_audio_enc = settings['lm']['freeze_audio_enc']
        if self.freeze_audio_enc:
            for p in self.audio_enc.parameters():
                p.requires_grad = False
        
      
        # Main model configuration
        self.bart_config = BartConfig(vocab_size=settings['lm']['config']['vocab_size'],
                                custom_vocab_size=settings['lm']['config']['custom_vocab_size'], #for using custom vocab size bart model
                                encoder_layers=settings['lm']['config']['encoder_layers'],
                                encoder_ffn_dim=settings['lm']['config']['encoder_ffn_dim'],
                                encoder_attention_heads=settings['lm']['config']['encoder_attention_heads'],
                                decoder_layers=settings['lm']['config']['decoder_layers'],
                                decoder_ffn_dim=settings['lm']['config']['decoder_ffn_dim'],
                                decoder_attention_heads=settings['lm']['config']['decoder_attention_heads'],
                                activation_function=settings['lm']['config']['activation_function'],
                                d_model=settings['lm']['config']['d_model'],
                                dropout=settings['lm']['config']['dropout'],
                                attention_dropout=settings['lm']['config']['attention_dropout'],
                                activation_dropout=settings['lm']['config']['activation_dropout'],
                                classifier_dropout=settings['lm']['config']['classifier_dropout'],
                                max_length=settings['lm']['generation']['max_length'],
                                min_length=settings['lm']['generation']['min_length'],
                                early_stopping=settings['lm']['generation']['early_stopping'],
                                num_beams=settings['lm']['generation']['num_beams'],
                                length_penalty=settings['lm']['generation']['length_penalty'],
                                no_repeat_ngram_size=settings['lm']['generation']['no_repeat_ngram_size'])
        print(self.bart_config)
        
        # Other parameters
        audio_emb_size = settings['adapt']['audio_emb_size']
        lm_emb_size = self.bart_config.d_model
        pretrained_lm = settings['lm']['pretrained']
        n_adapt_layers = settings['adapt']['nb_layers']
        tokenizer =  settings['lm']['tokenizer']   
        
        # Audio features to d_model embeddings
        if n_adapt_layers >= 1:
            audio_adapt_list = [nn.Linear(audio_emb_size, lm_emb_size)]
            for i_adapt in range(n_adapt_layers-1):
                audio_adapt_list.append(nn.ReLU(inplace=True))
                audio_adapt_list.append(nn.Linear(lm_emb_size, lm_emb_size))
            self.audio_adapt = nn.Sequential(*audio_adapt_list)
        else:
            self.audio_adapt = None
        
        if pretrained_lm is not None: # Bypass model configuration to load a pre-trained model (e.g. facebook/bart-base)
            # self.bart_lm = BartForConditionalGeneration.from_pretrained(pretrained_lm)
            
            # using custom vocab_embeddings BART model 
            self.bart_lm = self.load_bart_with_custom_embeddings(embedding_path = "./exp_settings", settings = settings, bart_model_name=pretrained_lm, freeze_embeddings=settings['lm']['config']['freeze_token_embedding'])
        else:
            if tokenizer == 'facebook/bart-base':
                self.bart_lm = BartForConditionalGeneration(self.bart_config)  
            else:
                self.bart_lm = self.load_bart_with_custom_embeddings(embedding_path = "./exp_settings", settings = settings, bart_model_name=pretrained_lm, freeze_embeddings=settings['lm']['config']['freeze_token_embedding'])


        # Freezing
        if settings['lm']['freeze']['all']:
            for p in self.bart_lm.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.encoder.embed_positions.parameters():
                p.requires_grad = True
            for p in self.bart_lm.model.encoder.layers[0].self_attn.parameters():
                p.requires_grad = True
        if settings['lm']['freeze']['dec']:
            for p in self.bart_lm.model.shared.parameters():
                p.requires_grad = False
            for p in self.bart_lm.model.decoder.parameters():
                p.requires_grad = False
            for p in self.bart_lm.lm_head.parameters():
                p.requires_grad = False
        if settings['lm']['freeze']['enc']:
            for p in self.bart_lm.model.encoder.parameters():
                p.requires_grad = False
        if settings['lm']['freeze']['attn']:
            for l in self.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['mlp']:
            for l in self.bart_lm.modules():
                if isinstance(l, Linear):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['dec_attn']:
            for l in self.bart_lm.model.decoder.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze']['dec_mlp']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['dec_self_attn']:
            for l in self.bart_lm.model.decoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['enc_mlp']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze']['enc_attn']:
            for l in self.bart_lm.model.encoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False

    
    # Custom implementation of the Bart forward function
    def forward(self,
                audio_features=None,
                cond_tokens=None,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None, #True
                output_hidden_states=None,
                return_dict=True,
                keyword_embeddings=None,
                file_name=None
        ):
        
        audio_embs = self.audio_enc.forward_frame_embeddings(audio_features)
        ################################ Passing Keyword Information ######################################
        
        # topk = 5
        # batch_probs = pretrained_audio_enc(audio_features)["clipwise_output"]

        # batch_keyword_embs = []
        # batch_extra_masks = []

        # for b in range(batch_probs.shape[0]):
        #     probs = batch_probs[b]  # shape: [num_classes]

        #     # Get top-k indices
        #     values, indices = torch.topk(probs, k=topk)

        #     # Map indices → embeddings
        #     topk_embeddings = [top5_classes_embeddings[idx.item()] for idx in indices]
        #     keyword_embeddings = torch.cat(topk_embeddings, dim=0)  # (k, 768)

        #     # Pad to 30
        #     num_keywords = keyword_embeddings.size(0)
        #     zero_padding = torch.zeros(30 - num_keywords, 768, device=keyword_embeddings.device)
        #     keyword_embeddings = torch.cat((keyword_embeddings, zero_padding), dim=0)  # (30, 768)

        #     # Attention mask for these 30 tokens
        #     extra_mask = torch.zeros(30, dtype=attention_mask.dtype, device=attention_mask.device)
        #     extra_mask[:num_keywords] = 1

        #     batch_keyword_embs.append(keyword_embeddings.unsqueeze(0))  # (1, 30, 768)
        #     batch_extra_masks.append(extra_mask.unsqueeze(0))           # (1, 30)

 
        # # Stack all batches
        # batch_keyword_embs = torch.cat(batch_keyword_embs, dim=0).to(device)  # (B, 30, 768)
        # batch_extra_masks = torch.cat(batch_extra_masks, dim=0).to(device)    # (B, 30)

        # # Concatenate audio + keywords
        # audio_embs = torch.cat((audio_embs, batch_keyword_embs), dim=1)  # (B, T+30, 768)
        # attention_mask = torch.cat((attention_mask, batch_extra_masks), dim=1)  # (B, T+30)
    
        #################################################################################
 

        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=audio_embs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True)['last_hidden_state']
        
        encoder_outputs = [encoder_outputs]
        
        # Decoder-only pass
        outputs = self.bart_lm(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
          )
        
        return outputs['loss'], outputs['logits']
        
    def generate_greedy(self,
                audio_features=None,
                cond_tokens=None,
                attention_mask=None,
                inputs_embeds=None
        ):
        
        audio_embs = self.audio_enc(audio_features, skip_fc=True)
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
            audio_embs = audio_features
        
        encoder_outputs = self.bart_lm.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=audio_embs,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True)
        
        max_len = self.bart_lm.config.max_length
        cur_len = 0
        
        input_ids = torch.zeros((audio_embs.size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        
        outputs = self.bart_lm(input_ids=None,
                        attention_mask=attention_mask,
                        decoder_input_ids=input_ids,
                        decoder_attention_mask=None,
                        inputs_embeds=audio_embs,
                        use_cache=True,
                        return_dict=True)
        
        _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
        _past = outputs['past_key_values']
        _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
        input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)
        
        # Override with bos token
        input_ids[:, 1] = self.bart_lm.config.bos_token_id
        cur_len += 1
        
        while cur_len < max_len:
            model_inputs = self.bart_lm.prepare_inputs_for_generation(input_ids, past=_past, attention_mask=attention_mask, encoder_outputs=[encoder_outputs['last_hidden_state']])
            outputs = self.bart_lm(**model_inputs)
            _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
            _past = outputs['past_key_values']
            _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
            input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)
            cur_len += 1
        
        #print(input_ids)
        return input_ids
    
    def generate_beam(self,
                  audio_features=None,
                  cond_tokens=None,
                  attention_mask=None,
                  file_name=None,  # list of file names in batch
                  inputs_embeds=None,
                  output_attentions=None,
                  keyword_embeddings=None):

        self.bart_lm.force_bos_token_to_be_generated = True

        audio_embs = self.audio_enc.forward_frame_embeddings(audio_features)

    #################################### Passing Keyword Information ###########################################
    
        # topk = 5
        # batch_probs = pretrained_audio_enc(audio_features)["clipwise_output"]

        # batch_keyword_embs = []
        # batch_extra_masks = []

        # for b in range(batch_probs.shape[0]):
        #     probs = batch_probs[b]  # shape: [num_classes]

        #     # Get top-k indices
        #     values, indices = torch.topk(probs, k=topk)
        #     
        # #     # Map indices → embeddings
        #     topk_embeddings = [top5_classes_embeddings[idx.item()] for idx in indices]
        #     keyword_embeddings = torch.cat(topk_embeddings, dim=0)  # (k, 768)

        #     # Pad to 30
        #     num_keywords = keyword_embeddings.size(0)
        #     zero_padding = torch.zeros(30 - num_keywords, 768, device=keyword_embeddings.device)
        #     keyword_embeddings = torch.cat((keyword_embeddings, zero_padding), dim=0)  # (30, 768)

        #     # Attention mask for these 30 tokens
        #     extra_mask = torch.zeros(30, dtype=attention_mask.dtype, device=attention_mask.device)
        #     extra_mask[:num_keywords] = 1

        #     batch_keyword_embs.append(keyword_embeddings.unsqueeze(0))  # (1, 30, 768)
        #     batch_extra_masks.append(extra_mask.unsqueeze(0))           # (1, 30)

 
        # # Stack all batches
        # batch_keyword_embs = torch.cat(batch_keyword_embs, dim=0).to(device)  # (B, 30, 768)
        # batch_extra_masks = torch.cat(batch_extra_masks, dim=0).to(device)    # (B, 30)

        # # Concatenate audio + keywords
        # audio_embs = torch.cat((audio_embs, batch_keyword_embs), dim=1)  # (B, T+30, 768)
        # attention_mask = torch.cat((attention_mask, batch_extra_masks), dim=1)  # (B, T+30)
    
    ####################################################################################################
        
        

        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_embs)
        else:
           audio_embs = audio_features

       
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=audio_embs,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True)
        
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.bos_token_id
        # input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        # Beam decoding
        
        outputs = self.bart_lm.generate(
            attention_mask=attention_mask,   
            decoder_input_ids=input_ids,   
            decoder_attention_mask=decoder_attention_mask,  
            encoder_outputs=encoder_outputs,  
            output_attentions=output_attentions,
            return_dict_in_generate = False,
            use_cache=True)
       
        
        return outputs
        
    def load_bart_with_custom_embeddings(self, embedding_path, settings, bart_model_name="facebook/bart-base", freeze_embeddings=True):
       
        if bart_model_name is not None:
            model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        else:
            model = BartForConditionalGeneration(self.bart_config)
            print("Created custom BART model")
            
        
        # Load custom embeddings
        custom_embeddings = torch.load(os.path.join(embedding_path, 'custom_bart_embeddings.pt'))
        
        # Update both encoder and decoder embeddings
        model.model.encoder.embed_tokens.weight.data = custom_embeddings
        model.model.decoder.embed_tokens.weight.data = custom_embeddings
        
        # If using shared embeddings, this line is also necessary
        if model.config.tie_word_embeddings:
            model.model.shared.weight.data = custom_embeddings

        if freeze_embeddings:
            # Freeze encoder embeddings
            model.model.encoder.embed_tokens.weight.requires_grad = False
            
            # Freeze decoder embeddings
            model.model.decoder.embed_tokens.weight.requires_grad = False
            
            # Freeze shared embeddings if they exist
            if model.config.tie_word_embeddings:
                model.model.shared.weight.requires_grad = False

        # changing the last layer after BART decoder to custom vocab size 
        model.config.vocab_size = settings['lm']['config']['custom_vocab_size']

        model.lm_head = nn.Linear(self.bart_config.d_model, settings['lm']['config']['custom_vocab_size'])
        model.register_buffer("final_logits_bias", torch.zeros((1, settings['lm']['config']['custom_vocab_size'])))

        model.lm_head.weight.data.normal_(mean=0.0, std=self.bart_config.init_std)
        
        return model
