#!/usr/bin/python  
# -*- coding:utf-8 -*- 
from pprint import pprint
import json,xmltodict
namespaces={'article':None}
with open('small.xml','r') as f:
	data_json = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23
}

	# data_json = json.dumps(data)
	# print type(xmltodict.parse(f.read()))
	
	# print xmltodict.parse(f.read())
	data_json=json.dumps(xmltodict.parse(f.read()))#,process_namespaces=True, namespaces=namespaces, indent=4
	# # a=xmltodict.parse(f.read(),process_namespaces=True, namespaces=namespaces)
	# print type(json.loads(data)),type(data)
	# # print json.loads(data)
	with open('json.json', 'w+') as fo:
		# print data
		json.dump(data_json, fo)
	# with open('json.json','r') as ff:
	# 	j= json.load(ff)
	# 	print j
	# 	print json.loads(j),type(json.loads(j))
# import simplejson  

# 	# str ="<?xml version="1.0" ?><person><name>john</name><age>20</age></person"
# 	dic_xml = convert_to_dic(f.read())
# from __future__ import print_function
# import xml.sax
# import sys  
# import io
# import traceback 
# # sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码  
# class MovieHandler( xml.sax.ContentHandler ):
#    res=''
#    def __init__(self):
#       self.CurrentData = ""
#       self.author = ""
#       self.title = ""
#       self.year = ""
#       self.journal = ""
#    # 元素开始事件处理
#    def startElement(self, tag, attributes):
#       self.CurrentData = tag
#       if tag == "article":
#          print("self.__class__.res=",self.__class__.res)
#          try:
#            ww.write(self.__class__.res+'\n')
#          except:
#              traceback.print_exc()
#          self.__class__.res=''
#         # print ("*****article*****")
#          mdate = attributes["mdate"]
#          #print ("mdate:", mdate)
#          key=attributes["key"]
#          #print ("key:",key)
#          self.__class__.res=self.__class__.res
#          #print ('res_init:',self.__class__.res)

#    # 元素结束事件处理
#    def endElement(self, tag):
#       if self.CurrentData == "author":
#          #print ("author:", self.author)
#          self.__class__.res=self.__class__.res+self.author+'|| '
#       elif self.CurrentData == "title":
#          #print ("title:", self.title)
#          self.__class__.res=self.__class__.res+self.title+'|| '
#       elif self.CurrentData == "year":
#          #print ("year:", self.year)
#          self.__class__.res=self.__class__.res+self.year+'|| '
#       elif self.CurrentData == "journal":
#          #print ("journal:", self.journal)
#          self.__class__.res=self.__class__.res+self.journal+'|| '
#       self.CurrentData = ""


#    # 内容事件处理
#    def characters(self, content):
#       if self.CurrentData == "author":
#          self.author = content+'&&'
#       elif self.CurrentData == "title":
#          self.title = content
#       elif self.CurrentData == "year":
#          self.year = content
#       elif self.CurrentData == "journal":
#          self.journal = content
# if ( __name__ == "__main__"):

#    parser = xml.sax.make_parser()
#    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
#    Handler = MovieHandler()
#    parser.setContentHandler( Handler )
#    ww=open('simple.txt','w+')
#    parser.parse("small.xml")
#    ww.close()





















# import lxml
# from lxml import etree
# infile = 'simple.xml'
# context = etree.iterparse(infile,events=('end',),tag='article')
# print context.text
# for event,elem in context:
# 	pass

# from lxml import etree
# class TitleTarget(object):
# 	def __init__(self):
# 		self.text = []
# 	def start(self, tag, attrib):
# 		self.is_title = True if tag == 'dblp' else False
# 	def end(self, tag):
# 		pass
# 	def data(self, data):
# 		if self.is_title:
# 			self.text.append(data)
# 	def close(self):
# 		return self

# parser=etree.XMLParser(target=TitleTarget())
# # This and most other samples read in the Google copyright data
# infile = 'simple.xml'
# results = etree.parse(infile, parser)
# print results.text
# When iterated over, 'results' will contain the output from 
# target parser's close() method
# out = open('titles.txt', 'w')
# out.write('\n'.join(results))
# out.close()