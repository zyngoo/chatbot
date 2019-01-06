# -*- coding:utf-8 -*-
import argparse
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pprint import pprint
# from chatbot import ChatBot

import os
import re
import random
import time

import jieba
import tensorflow as tf

from models import model_helper
from models.attention_model import AttentionModel
from models.basic_model import BasicModel
from utils import misc_utils as utils
from utils.vocabulary import Vocabulary
from utils import param_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
param_utils.add_arguments(parser)
FLAGS, unused = parser.parse_known_args()

hparams = param_utils.create_hparams(FLAGS)

hparams = param_utils.create_hparams(FLAGS)

json_str = open('hparams/chatbot_xhj.json').read()
loaded_hparams = param_utils.create_hparams(FLAGS)
loaded_hparams.parse_json(json_str)

param_utils.combine_hparams(hparams, loaded_hparams)



# Warm up jieba
jieba.lcut("jieba")

class ChatBot:

    def __init__(self, hparams):
        self.hparams = hparams
        # print("====test__init__==\n")
        # Data locations
        self.out_dir = hparams.out_dir
        # print("our_dir:", self.out_dir)
        self.model_dir = os.path.join(self.out_dir, 'ckpts')
        # print("model_dir:", self.model_dir)
        # Create models
        attention_option = hparams.attention_option

        if attention_option:
            model_creator = AttentionModel
        else:
            model_creator = BasicModel

        self.infer_model = model_helper.create_infer_model(
            hparams=hparams,
            model_creator=model_creator)

        # Sessions
        config_proto = utils.get_config_proto()
        self.infer_sess = tf.Session(config=config_proto, graph=self.infer_model.graph)

        # EOS
        self.tgt_eos = Vocabulary.EOS.encode("utf-8")
        # Load infer model
        with self.infer_model.graph.as_default():
            self.loaded_infer_model, self.global_step = model_helper.create_or_load_model(
                self.infer_model.model, self.model_dir, self.infer_sess, "infer")


    def chat(self, question):
        """Accept a input str and get response by trained model."""
        infer_model = self.infer_model
        infer_sess = self.infer_sess
        beam_width = self.hparams.beam_width

        input_seg = jieba.lcut(question)
        # print("input_seg = ", input_seg)
        iterator_feed_dict = {
            infer_model.src_data_placeholder: input_seg,
            infer_model.batch_size_placeholder: 1
        }
        infer_sess.run(
            self.infer_model.iterator.initializer,
            feed_dict=iterator_feed_dict)

        sample_words = self.loaded_infer_model.decode(infer_sess)

        if beam_width > 0:
            # Get a random answer.
            beam_id = random.randint(0, beam_width - 1)
            sample_words = sample_words[beam_id]

        response = self._get_response(sample_words)

        return response

    def _get_response(self, sample_words):
        tgt_eos = self.tgt_eos
        # Make sure sample_words has 1 dim.
        sample_words = sample_words.flatten().tolist()

        if tgt_eos and tgt_eos in sample_words:
            sample_words = sample_words[:sample_words.index(tgt_eos)]

        response = ' '.join([word.decode() for word in sample_words])

        return response


#ChatRobot界面类
class ChatRobotGUI(QDialog):

    def __init__(self):

        self.chatbot = ChatBot(hparams)

        QDialog.__init__(self)
        self.setWindowIcon(QIcon('./images/ico.png'))        #窗口左上方图标
        self.setFont(QFont('幼圆', 12))              #窗口内字体
        self.setWindowTitle('智能聊天机器人')         #窗口标题

        # 窗口内布局
        layout = QVBoxLayout()

        self.label_1 = QLabel('正在和小通聊天中')
        self.text_edit = QTextEdit()

        self.label_2 = QLabel('我：')
        self.line_edit = QLineEdit()
        self.setFocus()

        self.button_1 = QPushButton('发送')
        self.button_2 = QPushButton('关闭')

        #布局中加入控件
        layout.addWidget(self.label_1)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.label_2)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button_1)
        layout.addWidget(self.button_2)

        self.setLayout(layout)             #设置布局

        self.button_1.clicked.connect(self.showText)        #鼠标点击发送事件时发生showText()方法
        self.button_2.clicked.connect(self.close)           #鼠标点击关闭事件时发生close()方法


    #获取方法，得到文本框内的内容以及ChatRobot所回复的内容
    def getText(self):
        # text = '我：' + '\n' + self.line_edit.text() + '\n\n' + '小黑：' + '\n' + self.chatbot.chat()
        text = '我：' + '\n' + self.line_edit.text() + '\n\n' + '小通：'
        return text

    def getAI(self):
        start_time = time.time()
        question = self.line_edit.text()
        # print("question = ", question)
        if question == "":
            question = "你猜"
        print("me > ", question)
        answer = self.chatbot.chat(question)
        answer = "".join(re.split(" ", answer))
        print("AI > %s (%.4fs)" % (answer, time.time() - start_time))

        text =  '\n' + answer
        return text

    #显示方法，将获取到的内容显示在文本域中
    def showText(self):

        self.text_edit.setText(self.getText() + self.getAI())
        self.line_edit.setText("")



