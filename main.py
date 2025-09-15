import torch
from pathlib import Path
import argparse
import shutil
from utils.file_io import load_yaml_file
from data_loader import get_dataset, default_data_collator
import yaml
from transformers import AutoTokenizer, TrainingArguments, BartTokenizer

from models import *
from trainer import *

def main(config):
    # Experiment settings
    dataset_path = config.dataset_path

    with open('./exp_settings/dcb.yaml') as f:
        settings = yaml.safe_load(f)

    # Override dataset_path if given from terminal
    if config.dataset_path is not None:
        settings['data']['root_dir'] = config.dataset_path

    print("Dataset path being used:", settings['data']['root_dir'])

    if isinstance(settings['training']['seed'], int):
        torch.manual_seed(settings['training']['seed'])

    if torch.cuda.is_available() and not settings['training']['force_cpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Training arguments
    out_dir = './outputs/'+config.exp+'_out'
    dataset= config.dataset_name
    

    # Save YAML file for each experiment
    Path(out_dir).mkdir(parents=True, exist_ok=True) # Create output directory
    destination_yaml = Path(out_dir) / Path('./exp_settings/dcb.yaml').name 
    shutil.copy(Path('./exp_settings/dcb.yaml'), destination_yaml)  # Copy the file to the directory

    print(f"YAML file saved at: {destination_yaml}")

    training_args = TrainingArguments(output_dir=out_dir,
                    learning_rate=settings['training']['lr'],
                    per_device_train_batch_size=settings['training']['batch_size'],
                    per_device_eval_batch_size=settings['evaluation']['batch_size'],
                    gradient_accumulation_steps=settings['training']['gradient_accumulation_steps'],
                    dataloader_num_workers=settings['training']['num_workers'],
                    save_steps=settings['training']['save_steps'],
                    num_train_epochs=float(settings['training']['nb_epochs']),
                    evaluation_strategy='steps' if settings['workflow']['validate'] else 'no',
                    eval_steps=settings['training']['eval_steps'],
                    load_best_model_at_end=True if settings['workflow']['validate'] else False)
    #print(training_args)
    
    # Tokenizer
    if settings['lm']['tokenizer'] == 'facebook/bart-base':
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base', use_fast=True)

    else:
         #Using custom tokenizer
        tokenizer = BartTokenizer.from_pretrained("settings['lm']['tokenizer']", use_fast=True)

    
    # Datasets
    data_train = None
    data_eval = None
    if settings['workflow']['train']:
        data_train, data_eval = get_dataset('training', settings, tokenizer,dataset )
        print('Loaded development dataset.')
    
    # Model
    model = BARTAAC(settings, device)
     
    
    print(model)
    print('Num. parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('Num. trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad==True)))
    
    # Trainer
    trainer = BARTAACTrainer(model, args=training_args, data_collator=default_data_collator, train_dataset=data_train, eval_dataset=data_eval)
    
    # Workflow
    if settings['workflow']['train']:
        trainer.train()
        # Save best model state_dict, which is loaded at the end of training
        torch.save(trainer.model.state_dict(), out_dir+'/pytorch_model_best.bin')
        
    if settings['workflow']['evaluate'] or settings['workflow']['infer']:
        # Load model state_dict
        if settings['lm']['eval_model'] == 'checkpoint': # Specific checkpoint
            model.load_state_dict(torch.load(out_dir+'/checkpoint-'+str(settings['lm']['eval_checkpoint'])+'/pytorch_model.bin', map_location=device))
            print('Loaded model from checkpoint {}.'.format(settings['lm']['eval_checkpoint']))
        elif settings['lm']['eval_model'] == 'best': # Best validation loss model
            state_dict = torch.load(out_dir+'/pytorch_model_best.bin', map_location=device)
             
            model.load_state_dict(torch.load(out_dir+'/pytorch_model_best.bin', map_location=device))
            print('Loaded best validation loss model.')
        else: # Custom model weights, e.g. pre-trained weights. eval_model parameter should be /path/to/model.bin
            model.load_state_dict(torch.load(settings['lm']['eval_model'], map_location=device))
            print('Loaded custom model weights from {}.'.format(settings['lm']['eval_model']))
        model.bart_lm.config.force_bos_token_to_be_generated = True
    if settings['workflow']['evaluate']:
        if dataset == 'Clotho':
            data_eval, _ = get_dataset('evaluation', settings, tokenizer,dataset)   
            print('Loaded evaluation dataset.')
        else: 
            data_eval, _ = get_dataset('test', settings, tokenizer,dataset)  # for audiocaps its test
            print('Loaded evaluation dataset.')

        trainer.args.remove_unused_columns = False # Keep file_name key
        trainer.caption_evaluate(data_eval, tokenizer, generation_mode=settings['lm']['generation']['decoding'])
     
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='exp001', help='Experiment settings YAML file')
    parser.add_argument('--dataset_name', type=str, default='Clotho')
    parser.add_argument('--dataset_path', type=str, default='data_32k_224_mels')

    config, unknown = parser.parse_known_args()  # <-- allows extra args
    print(config)
    main(config)
