# -*- coding: utf-8 -*-
#!/usr/bin/env python
#@date  :2015-3-from
import tornado.httpserver
import tornado.ioloop
import tornado.web
import os
from tornado.options import define, options
from main import main
import json
from pprint import pprint


define("port", default=8888, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html', state=1)
class MatchGraphHandler(tornado.web.RequestHandler):
    def post(self):
        print 'post message'
        data= tornado.escape.json_decode(self.request.body)
        if not  data['k']:
            data['k'] = 4
        print data
        ret = main(graph_path=data['g'], query_graph=data['q'], k=data['k'], filterFlag=data['filterFlag'], commend=0)
        ret = json.dumps(ret,ensure_ascii=False, indent=2)
        print 'ret: ',ret
        self.write(ret)
    # def post(self):
    #     print 'post message'
    #     data= tornado.escape.json_decode(self.request.body)
    #     if not  data['k']:
    #         data['k'] = 4
    #     print data
    #     ret = main(graph_path=data['g'], query_graph=data['q'], k=data['k'], filterFlag=data['filterFlag'], commend=0)
    #     ret = json.dumps(ret,ensure_ascii=False, indent=2)
    #     print 'ret: ',ret
    #     self.write(ret)

class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r'/', IndexHandler),
            (r'/match_graph',MatchGraphHandler),
        ]
        settings = dict(
            cookie_secret="7CA71A57B571B5AEAC5E64C6042415DE",
            debug=True

        )

        tornado.web.Application.__init__(self, handlers, **settings)


if __name__ == "__main__":
    tornado.options.parse_command_line()
    Application().listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
