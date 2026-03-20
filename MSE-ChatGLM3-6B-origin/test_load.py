import argparse
from data.load_data import MMDataLoader
from config.config_classification import ConfigClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=str, default="classification")
    parser.add_argument('--modelName', type=str, default='cmcm')
    parser.add_argument('--datasetName', type=str, default='iemocap4')
    parser.add_argument('--root_dataset_dir', type=str, default=r'd:\ProjectFiles\exp_202603\codefiles\MSE-Adapter-Expend\datasets')
    parser.add_argument('--pretrain_LM', type=str, default='ZhipuAI/chatglm3-6b') # modelscope id or local path
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    
    args = parser.parse_args()
    
    # Apply config
    config = ConfigClassification(args)
    args = config.get_config()
    
    # We will test DataLoader
    print(f"Loading {args.datasetName}...")
    try:
        dataloader = MMDataLoader(args)
        train_loader = dataloader['train']
        print(f"Train loader size: {len(train_loader.dataset)}")
        
        # Get first batch
        for i, batch in enumerate(train_loader):
            print(f"\n--- Batch {i} ---")
            print(f"Audio shape: {batch['audio'].shape}")
            print(f"Video shape: {batch['vision'].shape}")
            print(f"Text token shape: {batch['text'].shape}")
            print(f"Labels: {batch['labels']}")
            print(f"Raw Text examples: {batch['raw_text']}")
            break
            
        print("\n✅ DataLoader successfully verified!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
