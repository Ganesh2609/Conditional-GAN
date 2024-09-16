import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def gradient_penalty(discriminator:torch.nn.Module,
                     labels:torch.tensor,
                     real:torch.tensor,
                     fake:torch.tensor,
                     device:torch.device):
    
    N, C, W, H = real.shape
    epsilon = torch.rand(N, 1, 1, 1).repeat(1, C, W, H).to(device)
    interpolated_img = epsilon*real + (1-epsilon)*fake
    mixed_score = discriminator(interpolated_img, labels)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_img,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

def train_generator(generator:torch.nn.Module,
                    discriminator:torch.nn.Module,
                    batch:tuple,
                    optimizer:torch.optim.Optimizer,
                    scaler:torch.cuda.amp.GradScaler,
                    BATCH_SIZE:int,
                    Z_DIM:int,
                    device:torch.device):
    
    generator.train()
    discriminator.eval()
    
    real_img, labels = batch
    labels = labels.to(device)
    
    with torch.cuda.amp.autocast(enabled=False):
        noise = torch.randn(real_img.shape[0], Z_DIM, 1, 1, device=device)
        fake_img = generator(noise, labels)
        pred_probs = discriminator(fake_img, labels)
        loss = -torch.mean(pred_probs)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    del noise, fake_img, pred_probs
    return loss.item()


def train_discriminator(generator:torch.nn.Module,
                        discriminator:torch.nn.Module,
                        batch:tuple,
                        optimizer:torch.optim.Optimizer,
                        scaler:torch.cuda.amp.GradScaler,
                        BATCH_SIZE:int,
                        Z_DIM:int,
                        ITERATIONS:int,
                        LAMBDA_GP:int,
                        device:torch.device):
    
    generator.eval()
    discriminator.train()
    
    real_img, labels = batch
    real_img, labels = real_img.to(device), labels.to(device)
    
    for i in range(ITERATIONS):     
        
        with torch.cuda.amp.autocast(enabled=False):  
            noise = torch.randn(real_img.shape[0], Z_DIM, 1, 1, device=device)
            fake_img = generator(noise, labels)
            
            real_probs = discriminator(real_img, labels)
            fake_probs = discriminator(fake_img, labels)
            real_score = torch.mean(real_probs)
            fake_score = torch.mean(fake_probs)
        
            gp = gradient_penalty(discriminator=discriminator, labels=labels, real=real_img, fake=fake_img, device=device)
            loss = fake_score - real_score + LAMBDA_GP*gp
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    del real_img, fake_img, real_probs, fake_probs
    return -loss.item()


def train_models(generator:torch.nn.Module,
                 discriminator:torch.nn.Module,
                 dataloader:torch.utils.data.DataLoader,
                 gen_optimizer:torch.optim.Optimizer,
                 disc_optimizer:torch.optim.Optimizer,
                 gen_scaler:torch.cuda.amp.GradScaler,
                 disc_scaler:torch.cuda.amp.GradScaler,
                 BATCH_SIZE:int,
                 Z_DIM:int,
                 NUM_EPOCHS:int,
                 device:torch.device,
                 DISC_ITERATIONS:int=5,
                 LAMBDA_GP:int=10,
                 gen_path:str=None,
                 disc_path:str=None,
                 result_path:str=None):
    
    results = {
        'Generator Loss' : [],
        'Discriminator Loss' : []
    }
    
    
    for epoch in range(2, NUM_EPOCHS+2):

        disc_loss = 0   
        gen_loss = 0   
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, batch in t:
                disc_batch_loss  =  train_discriminator(generator=generator, 
                                                        discriminator=discriminator, 
                                                        batch=batch, 
                                                        optimizer=disc_optimizer, 
                                                        scaler=disc_scaler,
                                                        BATCH_SIZE=BATCH_SIZE, 
                                                        Z_DIM=Z_DIM,
                                                        ITERATIONS=DISC_ITERATIONS,
                                                        LAMBDA_GP=LAMBDA_GP,
                                                        device=device)
                
                gen_batch_loss = train_generator(generator=generator,
                                                 discriminator=discriminator,
                                                 scaler=gen_scaler,
                                                 batch=batch,
                                                 optimizer=gen_optimizer,
                                                 BATCH_SIZE=BATCH_SIZE,
                                                 Z_DIM=Z_DIM,
                                                 device=device)
                
                disc_loss += disc_batch_loss
                gen_loss += gen_batch_loss

                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS+1}] ')
                t.set_postfix({
                    'Gen Batch Loss' : gen_batch_loss,
                    'Gen Loss' : gen_loss/(i+1),
                    'Disc Batch Loss' : disc_batch_loss,
                    'Disc Loss' : disc_loss/(i+1),
                })
                
                if gen_path:
                    torch.save(obj=generator.state_dict(), f=gen_path)
                if disc_path:
                    torch.save(obj=discriminator.state_dict(), f=disc_path)
                
                # Save results every 100 batches
                if i % 100 == 0 and result_path:
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}.png'
                    generator.eval()
                    discriminator.eval()
                    
                    real_img, labels = batch
                    labels = labels.to(device)
                    
                    with torch.inference_mode():
                        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
                        fake_img = generator(noise, labels).cpu()
                    fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)
                    
                    fig, axis = plt.subplots(4, 8, figsize=(16, 8))

                    # Plot fake images 
                    idx = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]
                    j = 0
                    for i in idx:
                        ax = axis.flat[i]
                        ax.imshow(fake_img[j].permute(1, 2, 0), cmap='gray')
                        j+=1
                        ax.axis(False)

                    # Plot real images 
                    idx = [4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]
                    j=0
                    for i in idx:
                        ax = axis.flat[i]  
                        ax.imshow(real_img[j].permute(1, 2, 0), cmap='gray')
                        j+=1
                        ax.axis(False)

                    plt.tight_layout()
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)

                
        results['Generator Loss'].append(gen_loss/len(dataloader))
        results['Discriminator Loss'].append(disc_loss/len(dataloader))
        
    return results