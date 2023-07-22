## Image Captioning - Deep Learning
### Dataset
<a href="https://www.kaggle.com/datasets/adityajn105/flickr8k?select=Images">Flickr 8k Dataset</a>

Pretrained models were used for tokenization: bert-base-cased

### Models
Image encoder: using viT_pretrained_model - google/vit-base-patch16-224

Caption encoding: using AutoModel.from_pretrained('bert-base-cased', config=bert_config)

Decoder: TransformerDecoder


### The model's performance and issues during training

* During the training process, both train loss and test loss decreased rapidly, and the model was often able to generate correct captions during the test() function by epoch 3. To assess the model's similarity score, Bleu was used. Additionally, the test loss was monitored to observe its decreasing pattern, and after each epoch, the predicted captions were printed using the test() function.
  
  To address the possibility of overfitting, a demo program was created to visualize the generated captions for unseen images using the checkpoints from each epoch. However, the results revealed that no captions were produced.
  
  Despite the model's excellent performance in terms of loss and Bleu score, the inability to generate captions during the demo program indicates a potential issue that needs to be investigated further.
* To eliminate the possibility of the model not training successfully due to overfitting, I attempted to train the model with different dropout values (0.3, 0.4, 0.5). Despite varying the dropout rate, the test() function was still capable of generating perfect captions by epoch 3. However, when using the demo() function on unseen images, no captions were generated.
  
  This suggests that the issue might not be related to overfitting or the dropout rate but could be attributed to other factors affecting the model's performance during the generation of captions for unseen images. Further investigation is required to identify the root cause of the problem.
* I made a modification to the decoder model's structure: in the forward function, I rearranged the order of operations. Instead of converting the encoded image and encoded ids' hidden dimensions to the decoder dim before fusion, I first fused the encoded image and encoded ids, and then converted them to the decoder dim. After making this adjustment, the demo() function was able to generate captions for unseen images successfully.
  ```
  combined_features = encoded_images + bert_embeddings      # (batch_size, sequence_length, encoder_dim)
  combined_features = self.fc_combined(combined_features)   # (batch_size, 40, 768)
  combined_features = self.activation(combined_features)
  encoded_images = self.fc_image(encoded_images)
  ```
* Next, I retrained the model and tried various combinations and parameters, I will list three of the most representative ones:

  Tpye 1:
  ```
  batch_size = 100
  lr = 0.00003        
  encoder_dim = 768   
  decoder_dim = 512   
  d_model = 512       
  nhead = 8           
  num_layers = 6      
  num_epoch = 10
  ```
