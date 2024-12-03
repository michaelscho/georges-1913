
# Medieval Latin Normalization Model based on Georges 1913

This repository contains the implementation of a PyTorch-based **text2text model with attention** for normalizing orthographic variations in medieval Latin texts. 

The model is trained on the [**Normalized Georges 1913 Dataset**](https://huggingface.co/datasets/mschonhardt/georges-1913-normalization) and leverages Hugging Face's ecosystem for easy model and vocabulary management.

## **Contents**
- **`train_model.py`**: Script for training the normalization model.
  - Includes dynamic loading of the dataset and vocabulary.
  - Trains a Seq2Seq model with an attention mechanism and saves the model and vocabulary for later use.
- **`test_model.py`**: Script for testing the normalization model.
  - Loads the trained model, vocabulary, and configuration from a Hugging Face repository.
  - Normalizes test words from an input file (`test_normalisation.txt`).

## **Usage**
1. **Train the Model**:
   - Modify `train_model.py` as needed for your dataset.
   - Run:
     ```bash
     python train_model.py
     ```
   - Saves:
     - Model: `normalization_model.pth`
     - Vocabulary: `vocab.pkl`
     - Config: `config.json`

2. **Test the Model**:
   - Uses https://huggingface.co/mschonhardt/georges-1913-normalization-model as default. If training your own model, ensure the model, vocabulary, and configuration are uploaded to a Hugging Face repository.
   - Add words to `test_normalisation.txt` for testing.
   - Run:
     ```bash
     python test_model.py
     ```
   - Outputs the normalized forms of the test words.

## **Acknowledgments**
Dataset and model were created by Michael Schonhardt ([https://orcid.org/0000-0002-2750-1900](https://orcid.org/0000-0002-2750-1900)) for the project Burchards Dekret Digital.

Creation was made possible thanks to the lemmata from Georges 1913, kindly provided via [www.zeno.org](http://www.zeno.org/georges-1913) by 'Henricus - Edition Deutsche Klassik GmbH'. Please consider using and supporting this valuable service.

## **License**
CC BY 4.0 ([https://creativecommons.org/licenses/by/4.0/legalcode.en](https://creativecommons.org/licenses/by/4.0/legalcode.en))