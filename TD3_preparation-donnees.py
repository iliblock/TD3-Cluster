# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:34:00 2024

@author: iliana
"""

import csv   
import json
import glob

def creerjson(nom,struc):
    nombre = str(nom)
    with open(f'{nombre}.json', "w", encoding = 'utf-8') as w:
        json.dump(struc, w, indent = 2, ensure_ascii = False)


dico_listes = {}

fichiers = glob.glob("**/*.bio", recursive = True)
# print(fichiers)
for chemin in fichiers:
    partie = chemin.split("\\")
    # print(chemin, "********************", partie)
    nom = partie[-2]
    # print(nom)

    lihta = []
    with open(chemin,newline='',encoding = 'utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', quotechar = "\n")
        for el in csv_reader:    
            lihta.append(el)
        # print(lihta)
    
    
    lis2 = []
        
    for elm in lihta:
        # print(len(elm))
        if len(elm) > 1:
            # print("OK",chemin)
            if elm[1] != 'O' and elm[0] != '':
                lis2.append(elm[0])
    # print(lis2)
    
    dico_listes[nom] = lis2
    
# print(dico_sets)

creerjson("Dictionaire_listes", dico_listes)

