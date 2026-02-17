# High-Fidelity Neural Speech Reconstruction through an Efficient Acoustic-Linguistic Dual-Pathway Framework

## Demo running
If you need to one-key running (especially the linguistic pathway that generate speech from word token features), a demo link with actural neural data is available [here](https://www.dropbox.com/scl/fi/54a3tlykmge4ap78cd72g/Neural2speech2_linguistic_pathway_demo.zip?rlkey=g0xmean3pvs2dnutetolw3wt8&st=twyjnbnu&dl=0)

## Abstract
Reconstructing speech from neural recordings is crucial for understanding speech coding and developing brain-computer interfaces (BCIs). However, existing methods trade off acoustic richness (pitch, prosody) for linguistic intelligibility (words, phonemes). To overcome this limitation, we propose a dual-path framework to concurrently decode acoustic and linguistic representations. The acoustic pathway uses a long-short term memory (LSTM) decoder and a high-fidelity generative adversarial network (HiFi-GAN) to reconstruct spectrotemporal features. The linguistic pathway employs a transformer adaptor and text-to-speech (TTS) generator for word tokens. These two pathways merge via voice cloning to combine both acoustic and linguistic validity. Using only 20 minutes of electrocorticography (ECoG) data per subject, our approach achieves highly intelligible synthesized speech (mean opinion score = 3.956 ± 0.173, word error rate = 18.9% ± 3.3%). Our dual-path framework reconstructs natural and intelligible speech from ECoG, resolving the acoustic-linguistic trade-off.

## Demo Page of Speech Waveforms

The demo page is available [here](https://cctn-bci.github.io/Neural2Speech2/).

## Demo Video for Introducing Our Work

The demo video is available [here](./video/demovideo.mp4).
