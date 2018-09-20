# coding=utf-8
import os
import subprocess

import tornado.gen
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options

define("port", default=8000, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        self.render('index.html')


class AjaxHandler(tornado.web.RequestHandler):
    def post(self):
        topic_text = self.get_argument('message1')
        head_text = self.get_argument('message2')
        type_text = self.get_argument('message3')
        if topic_text == "":
            topic_text = "月下独酌".decode("utf-8")
        if head_text == "":
            head_text = "金樽对月空自赏".decode("utf-8")
        file_replace(topic_text, head_text, type_text)
        path = os.getcwd()
        if type_text == 'dlh' or type_text == 'ymr':
            cmd = 'python ' + path + '/twinLstmSong.py'
            fp = type_text + "_out_song.txt"
        else:
            cmd = 'python ' + path + '/twinLstmPoem.py'
            fp = "out_song.txt"
        child = subprocess.Popen(cmd, cwd=path, shell=True)
        child.wait()
        fo = open(fp, 'r')
        lines = fo.readlines()
        res = ""
        for line in lines:
            if '/' in line and 'input' not in line and 'origin' not in line:
                res += line
            else:
                if 'UNK' in line:
                    res += "UNK/" + line
        self.write(res)


def file_replace(topic, head, type):
    s_topic = "".join(map(lambda x: x + '/', topic))
    s_head = "".join(map(lambda x: x + '/', head))
    fo = open('config.txt', 'r+')
    lines = fo.readlines()
    lines[0] = "poem_type = " + type.encode('utf-8') + " // " + "\n"
    lines[4] = "input_sen = " + s_topic.encode('utf-8') + "\n"
    lines[5] = "head = " + s_head.encode('utf-8') + "\n"
    fo.close()
    fo = open('config.txt', 'w+')
    fo.writelines(lines)
    fo.close()


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', IndexHandler),
            (r"/test", AjaxHandler),
        ]

        settings = {
            'template_path': 'templates',
            'static_path': 'static'
        }

        tornado.web.Application.__init__(self, handlers, **settings)


def main():
    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.bind(options.port, address='0.0.0.0')
    server.start(0)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()
