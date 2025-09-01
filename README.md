# High-Fidelity Neural Speech Reconstruction through an Efficient Acoustic-Linguistic Dual-Pathway Framework

## Abstract
Reconstructing speech from neural recordings is crucial for understanding speech coding and developing brain-computer interfaces (BCIs). However, existing methods trade off acoustic richness (pitch, prosody) for linguistic intelligibility (words, phonemes). To overcome this limitation, we propose a dual-path framework to concurrently decode acoustic and linguistic representations. The acoustic pathway uses a long-short term memory (LSTM) decoder and a high-fidelity generative adversarial network (HiFi-GAN) to reconstruct spectrotemporal features. The linguistic pathway employs a transformer adaptor and text-to-speech (TTS) generator for word tokens. These two pathways merge via voice cloning to combine both acoustic and linguistic validity. Using only 20 minutes of electrocorticography (ECoG) data per subject, our approach achieves highly intelligible synthesized speech (mean opinion score = 3.956 ± 0.173, word error rate = 18.9% ± 3.3%). Our dual-path framework reconstructs natural and intelligible speech from ECoG, resolving the acoustic-linguistic trade-off.

## Demo Page of Speech Waveforms

The demo page is available [here](https://cctn-bci.github.io/Neural2Speech2/).

## Demo Page of Speech Waveforms

The demo page is available [here](https://cctn-bci.github.io/Neural2Speech2/).

## Demo Video for Introducing Our Work

<video controls width="100%">
    <source src="./video/demovideo.mp4" type="video/mp4">
</video>

