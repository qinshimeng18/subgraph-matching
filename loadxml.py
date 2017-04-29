#!/usr/bin/python
# -*- coding:utf-8 -*-
from pprint import pprint
import json
import xmltodict


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
# if __name__ == '__main__':
#     loadxml('simple.xml','json.json')
