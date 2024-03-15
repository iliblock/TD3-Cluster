#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:16:20 2022

@author: antonomaz
"""


import numpy as np #on importe ce qui nous permet de travailler avec des matrices
from sklearn.cluster import AffinityPropagation #ce qui nous permettra de creer les clusters
from sklearn.metrics import DistanceMetric #ce qui nous donne des outils de mesurement
from sklearn.feature_extraction.text import CountVectorizer #ce qui nous permet de creer des matrices avec les mots
import sklearn #ce qui nous permet d'utiliser des différents algorithmes
import json #ce qui nous permet de creer et lire des fichiers qui contiennent des structures python
import glob #ce qui nous permet de faire que python puisse aller chercher et lire dans les dossiers de l'ordinateur
import re #ce qui nous permet d'utiliser les expressions régulières
from collections import OrderedDict #nous permet de creer des dictionnaires qui mantiennent l'ordre dont les éléments on été ajoutés


def lire_fichier (chemin): #c'est la fonction pour lire un fichier json(c'est à dire extraire les informations de chaque structure)
    with open(chemin) as json_data: 
        texte =json.load(json_data)
    return texte

def nomfichier(chemin): #fonction pour récupérer le nom d'un fichier
    nomfich= chemin.split("\\")[-1] #pour séparer la chaine de charactères du nom du chemin selon "/" et ne prendre que le nom du fichier
    nomfich= nomfich.split(".") #pour séparer le nom du fichier selon "."
    nomfich= ("_").join([nomfich[0],nomfich[1]]) #pour substituer tout ce qui avant était un point avec un _
    return nomfich
    
#%%
# chemin_entree = lire_fichier() #il faut rajouter ici le chemin du fichier qui contient les données pré-préparées




for subcorpus in glob.glob(".\\NuevosJson"): #on crée une boucle qui parcourt chaque dossier dans le dossier principal
    # print("SUBCORPUS***",subcorpus)
    liste_nom_fichier =[] #on crée une liste pour stocker les noms de chaque fichier plus tard
    for path in glob.glob("%s/*.json"%subcorpus): #on crée une boucle qui parcourt chaque fichier json contenu dans le dossier actuel de la boucle principale
        # print("PATH*****",path)
        
        nom_fichier = nomfichier(path) #on utilise la fonction de récupération de nom d'un fichier pour sticker le nom du fichier actuel dans une variable
        # print(nom_fichier)
        liste=lire_fichier(path) #on lit le fichier actuel et on stocke le contenu dans une variable
        
        # print(liste)
        
#### FREQUENCE ########
        
        dic_mots={} #on initialise un dictionnaire
        i=0 #on assigne la valeur 0 à la variable i (qui va juste servir à compter les étapes qui ont été faites)
    
        
        for mot in liste: #on crée une boucle qui parcourt chaque mot dans le texte actuel
            
            if mot not in dic_mots: #on compte la fréquence d'apparition de chaque mot dans le texte; Si le mot est dans le dictionnaire on ajoute 1 à sa valeur dans le dictionnaire, sinon on l'ajoute au dictionnaire (valeur 1)
                dic_mots[mot] = 1
            else:
                dic_mots[mot] += 1
        
        i += 1 #on additionne 1 à la valeur de la variable i

        new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0])) #on assigne a une variable un dictionnaire ordonné (ordonné dans le même ordre ou les clés ont étés rentrés, c'est à dire l'ordre de lecture des fichiers dans ce cas).
        #c'est à dire que new_d est un dictionnaire rempli de listes de paires. Chaque liste inclue dedans est le la liste des paires mots/fréquences dans un texte (ordonnés par ordre croissant).
        freq=len(dic_mots.keys()) #on stocke dans une variable la quantité de mots dans le fichier actuel
        

        Set_00 = set(liste) #on crée un set avec les mots du fichier actuel (pour ne pas avoir de mots répétés)
        Liste_00 = list(Set_00) #on crée une liste avec le set (pour pouvoir opérer plus facilement avec)
        dic_output = {} #on initialise un dictionnaire et 2 listes
        liste_words=[]
        matrice=[]
        
        for l in Liste_00: #on itère sur tous les mots de cette liste
                
            if len(l)!=1:
                liste_words.append(l) #si le mot a plus d'une lettre on l'ajoute à une liste de mots
        

        try: #on utilise try pour tester le code suivant et chercher des choses sans que le programme bloque
            words = np.asarray(liste_words) #on transforme la liste de mots qui vient d'être crée en array (pour pouvoir créer des matrices avec)
            for w in words: #on itère sur chaque ligne(?) de l'array
                liste_vecteur=[]
            
                    
                for w2 in words: #on itère sur chaque mot(?) de l'array
                
                        V = CountVectorizer(ngram_range=(2,3), analyzer='char') #on prend des ngrammes de 2 et 3 caractères
                        X = V.fit_transform([w,w2]).toarray() #on transforme les données des ngrammes en un d'array
                        distance_tab1=sklearn.metrics.pairwise.cosine_distances(X) #on calcule la distance des ngrammes
                        liste_vecteur.append(distance_tab1[0][1]) #on rajoute l'information qu'on vient d'avoir dans une liste par paires
                    
                matrice.append(liste_vecteur) #on rajoute la liste à la matrice (une liste par ligne dans words)
            matrice_def=-1*np.array(matrice)
           
                  
            affprop = AffinityPropagation(affinity="precomputed", damping= 0.6, random_state = None) #on assigne à affprop la fonction de calcul d'affinité de propagation
    
            affprop.fit(matrice_def) #on calcule la similarité entre tous les points de la matrice
            for cluster_id in np.unique(affprop.labels_): #pour chaque centroïde (qui est contenu dans les labels de chaque point unique de affprop)
                exemplar = words[affprop.cluster_centers_indices_[cluster_id]] #exemplar devient le centroïde
                cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)]) #tout label qui équivaut au cluster-id (le centroïde), qui ne soit pas vide et qui soit unique est assigné à la variable cluster
                cluster_str = ", ".join(cluster) #on met ensemble tous les mots de cluster avec une virgule et un espace
                cluster_list = cluster_str.split(", ") #on crée une liste en séparant les mots par la virgule et l'espace
                            
                Id = "ID "+str(i) #on crée un id pour chaque centroïde avec le compteur préparé
                for cle, dic in new_d.items(): #pour chaque clé et valeur dans le dictionnaire des mots ordonnés avec les frequences
                    if cle == exemplar: #si la clé équivaut au centroïde
                        dic_output[Id] ={} #on crée une clé dans le dictionnaire de centroïdes finaux avec le id (compteur)
                        dic_output[Id]["Centroïde"] = exemplar #on rajoute le nom du centroïde au dictionnaire
                        dic_output[Id]["Freq. centroide"] = dic #on rajoute la frequence qui était contenue dans new_d au nouveau dictionnaire
                        dic_output[Id]["Termes"] = cluster_list #on rajoute la liste des mots de chaque centroïde
                
                i=i+1 #on augmente le compteur de 1
            #    print(dic_output)
            

        except :        
            print("**********Non OK***********", path) #s'il y a une erreur quelque part (une "exception") on fait un printe pour visualiser où elle est

    
            liste_nom_fichier.append(path) #on rajoute le chemin du fichier à la liste de fichiers pour creer une liste des fichiers fautifs??
            
            
            continue #on fait remarcher le programme jusqu'à la fin

#%%

print(dic_output)

    
