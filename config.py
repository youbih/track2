# *_*coding:utf-8 *_*
import os

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_BASE = os.path.normpath(os.path.join(_CONFIG_DIR, '..', 'models'))

AFFECTGPT_ROOT = _CONFIG_DIR
EMOTION_WHEEL_ROOT = os.path.join(_CONFIG_DIR, 'emotion_wheel')
OUTSIDE_WHEEL_MAPPING = os.path.join(EMOTION_WHEEL_ROOT, 'wheel_mapping.npz')
RESULT_ROOT = os.path.join(_CONFIG_DIR, 'output/results')

###########################################
## 所有模型的存储路径 [放在一个路径下]
###########################################
PATH_TO_LLM = {
    'Qwen25': os.path.join(_MODEL_BASE, 'qwen2.5-7b-instruct'),
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': os.path.join(_CONFIG_DIR, 'models/clip-vit-large-patch14'),
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  os.path.join(_CONFIG_DIR, 'models/chinese-hubert-large'),
}

PATH_TO_MLLM = {
    ## For Qwen-Audio
    'qwen-audio-chat':            os.path.join(_MODEL_BASE, 'qwen-audio-chat'),
    ## For SALMONN
    'salmonn_7b':                 os.path.join(_MODEL_BASE, 'salmonn_7b.pth'),
    'vicuna-7b-v1.5':             os.path.join(_MODEL_BASE, 'vicuna-7b-v1.5'),
    'BEATs':                      os.path.join(_MODEL_BASE, 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'),
    'whisper-large-v2':           os.path.join(_MODEL_BASE, 'whisper-large-v2'),
    ## For Video-ChatGPT
    'video_chatgpt-7B':           os.path.join(_MODEL_BASE, 'video_chatgpt-7B.bin'),
    'LLaVA-7B-Lightening-v1-1':   os.path.join(_MODEL_BASE, 'LLaVA-7B-Lightening-v1-1'),
    'clip-vit-large-patch14':     os.path.join(_MODEL_BASE, 'clip-vit-large-patch14'),
    ## For Video-LLaMA
    'llama-2-7b-chat-hf':         os.path.join(_MODEL_BASE, 'llama-2-7b-chat-hf'),
    'imagebind_huge':             os.path.join(_MODEL_BASE, 'imagebind_huge.pth'),
    'video_llama_vl':             os.path.join(_MODEL_BASE, 'VL_LLaMA_2_7B_Finetuned.pth'),
    'video_llama_al':             os.path.join(_MODEL_BASE, 'AL_LLaMA_2_7B_Finetuned.pth'),
    'blip2_pretrained_flant5xxl': os.path.join(_MODEL_BASE, 'blip2_pretrained_flant5xxl.pth'),
    'bert-base-uncased':          os.path.join(_MODEL_BASE, 'bert-base-uncased'),
    'eva_vit_g':                  os.path.join(_MODEL_BASE, 'eva_vit_g.pth'),
    ## For Chat-UniVi
    'Chat-UniVi':                 os.path.join(_MODEL_BASE, 'Chat-UniVi'),
    ## For LLaMA-VID
    'llama-vid':                  os.path.join(_MODEL_BASE, 'llama-vid-7b-full-224-video-fps-1'),
    ## For mPLUG-Owl
    'mplug-owl':                  os.path.join(_MODEL_BASE, 'mplug-owl-llama-7b-video'),
    ## For Otter
    'otter':                      os.path.join(_MODEL_BASE, 'OTTER-Video-LLaMA7B-DenseCaption'),
    ## For VideoChat
    'vicuna-7b-v0':               os.path.join(_MODEL_BASE, 'vicuna-7b-v0'),
    'videochat_7b':               os.path.join(_MODEL_BASE, 'videochat_7b.pth'),
    ## For VideoChat2
    'umt_l16_qformer':            os.path.join(_MODEL_BASE, 'umt_l16_qformer.pth'),
    'videochat2_7b_stage2':       os.path.join(_MODEL_BASE, 'videochat2_7b_stage2.pth'),
    'videochat2_7b_stage3':       os.path.join(_MODEL_BASE, 'videochat2_7b_stage3.pth'),
    ## For Video-LLaVA
    'Video-LLaVA':                os.path.join(_MODEL_BASE, 'Video-LLaVA-7B'),
}


###################################################
## 所有数据集的存储路径 [所有标签都在 MER2026 路径下]
###################################################
DATA_DIR = {
    'MER2026':          '/work/2025/liusiyu/Dataset2026/mer2026-dataset',
}
PATH_TO_RAW_AUDIO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'audio', 'audio_track2_train_human', 'audio'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'audio', 'audio_track2_train_mercaptionplus', 'audio'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'audio', 'audio_track1_track2_candidate', 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'video', 'video_track2_train_human', 'video'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'video', 'video_track2_train_mercaptionplus', 'video'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'video'),
}
PATH_TO_RAW_FACE = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'openface_face'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'openface_face'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'track2_train_human.csv'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'track2_train_mercaptionplus.csv'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'track2_test.csv') if os.path.exists(os.path.join(DATA_DIR['MER2026'], 'track2_test.csv')) else os.path.join(DATA_DIR['MER2026'], 'track2_train_human.csv'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
