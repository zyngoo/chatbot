# _*_ encode:utf-8_*_

import sys

from PyQt5.QtWidgets import QApplication
from ChatRobotGUI import ChatRobotGUI

#主方法

if __name__ == '__main__':
    app = QApplication(sys.argv)

    dialog = ChatRobotGUI()
    dialog.show()

    app.exec_()
