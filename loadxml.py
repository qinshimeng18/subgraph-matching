#!/usr/bin/python
# -*- coding:utf-8 -*-
from pprint import pprint
import json
import xmltodict
def cutxml(xml_out):
    # xml_in = 'small.xml'
    xml_in = '../dblp-2017-03-03.xml/dblp-2017-03-03.xml'
    xml_out = xml_out
    count = 0 
    start = 360000
    end = 398010
    flag = 0
    flag_end = 0 
    with open(xml_out,'w') as out:
        with open(xml_in,'r') as f:
            out.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
            out.write('<!DOCTYPE dblp SYSTEM "dblp-2016-10-01.dtd">\n')
            out.write('<dblp>')
            for line in f:
                if count >start and count <=end:
                    if not flag:
                        if line[0:8] =='<article':
                            flag =1
                    if flag:
                        out.write(line)
                if count >end :
                    # if not flag_end:
                    if line[0:9] == '</article':
                        out.write('</article>')
                        break
                    else:
                        out.write(line)
                count +=1
            out.write('</dblp>')
def loadxml(xml_in,json_out):
    with open(xml_in, 'r') as f:
        data_json = xmltodict.parse(f.read())
        with open(json_out, 'w+') as fo:
            json.dump(data_json, fo)
    #     for i in xmltodict.parse(f.read())['dblp']['article']:
    #         if i.has_key('author') and i.has_key('title'):
    #             if type(i['title']) == dict:
    #                 i['title'] = i['title']['#text']
	   #          if type(i['author']) == list:
	   #              vertices[i['title']] = {'category': 'paper', 'weight': 1}
	   #              for j in i['author']:
	   #                  vertices[j] = {'category': 'person', 'weight': 1}
	   #                  edges.append((i['title'], j))
	   #          else:
	   #              vertices[i['author']] = {'category': 'person', 'weight': 1}
	   #              vertices[i['title']] = {'category': 'paper', 'weight': 1}
	   #              edges.append((i['title'], i['author']))
    # print edges
    # print vertices
if __name__ == '__main__':
    xml_out = 'qian.xml'
    xml_out = './static/8k13k.xml'
    cutxml(xml_out)
    # xml = './static/8k13k.xml'
    xml = xml_out
    json_out = 'qian.json'
    json_out = './static/8k13k.json'
    loadxml(xml,json_out)
