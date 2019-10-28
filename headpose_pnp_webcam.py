# -*- coding: utf-8 -*-
"""
Created on Mon March 18 22:08:09 2018

@author: mihaela
"""

import cv2
import imutils
import numpy as np
import dlib
 
# Functie pentru crearea unei liste cu coordonate faciale
def landm2coord(trasaturi, dtype="int"):
    # initializarea listei cu cele 68 de coordonate
    coord = np.zeros((68, 2), dtype=dtype)
 
    # parcurgem cele 68 de coordonate si 
    # returnam coordonatele intr-o lista cu formatul (x,y)
    for i in range(0, 68):
        coord[i] = (trasaturi.part(i).x, trasaturi.part(i).y)
 
    # returnam lista cu coordonate
    return coord
 
# functia main
if __name__=="__main__":
    # se incarca detectorul facial bazat pe histograma gradientilor orientati
    detector_facial = dlib.get_frontal_face_detector()
 
    # se incarca predictorul de trasaturi faciale
    predictor_trasaturi = dlib.shape_predictor('C:/Users/mihaela/Downloads/deepgaze-master/deepgaze-master/etc/shape_predictor_68_face_landmarks.dat')
 
    # 0 means your default web cam
    vid = cv2.VideoCapture(0)
    
 
    while True:
        _,cadru = vid.read()
        trasaturi_cheie=[]
        #redimensionam cadrul
        cadru = imutils.resize(cadru, width=400)
 
        # convertim fiecare cadru RGB intr-o imagine gray
		# pentru a creste eficienta si scadea timpul de executie
        cadru_gri = cv2.cvtColor(cadru,cv2.COLOR_BGR2GRAY)
 
        # detectam fata din fiecare cadru
        limite_fata = detector_facial(cadru_gri,0)
 
        for (enum,fata) in enumerate(limite_fata):
            # desenam un patrat asupra fetei din imagine
            x = fata.left()
            y = fata.top()
            w = fata.right() - x
            h = fata.bottom() - y
            
            cv2.rectangle(cadru, (x,y), (x+w, y+h), (120,160,230),2)
 
            # Avand regiunea de interes a fetei
			# afisam trasaturile fetei
            trasaturi = predictor_trasaturi(cadru_gri, fata)
         
            # convertim lista de trasaturi intr-un vector
            trasaturi = landm2coord(trasaturi)
			# gasim trasaturile cheie pe care le vom folosi
			
            trasaturi_cheie.append(trasaturi[30]) #varful nasului
            trasaturi_cheie.append(trasaturi[8])	#varful barbiei
            trasaturi_cheie.append(trasaturi[36])	#coltul stang al ochiului
            trasaturi_cheie.append(trasaturi[45])	#coltul drept al ochiului
            trasaturi_cheie.append(trasaturi[48])	#coltul stang al gurii
            trasaturi_cheie.append(trasaturi[54])	#coltul drept al gurii

            for (a,b) in trasaturi_cheie:
                # Afisam cele 6 trasaturi
                cv2.circle(cadru, (a, b), 2, (255, 0, 0), -1)
            puncte_2D = np.asarray(trasaturi_cheie, dtype=np.float32).reshape((6, 2))
            print (puncte_2D)
#            puncte_3D = np.array([
#                            (0.0, 0.0, 0.0),             # varful nasului
#                            (0.0, -330.0, -65.0),        # barbie
#                            (-225.0, 170.0, -135.0),     # coltul stang al ochiului
#                            (225.0, 170.0, -135.0),      # coltul drept al ochiului
#                            (-150.0, -150.0, -125.0),    # coltul stang al gurii
#                            (150.0, -150.0, -125.0)      # coltul drept al gurii
#                         
#                        ])
            puncte_3D = np.array([
                            (0.0, 0.0, 0.0),             # varful nasului
                            (3, 41, -88.0515),        # barbie
                            (-27, -15, -91.90),     # coltul stang al ochiului
                            (25, -17, -91.23),      # coltul drept al ochiului
                            (-13, 20, -88.9185),    # coltul stang al gurii
                            (16, 19, -81.5105)      # coltul drept al gurii
                        
                        ])
#            puncte_3D = np.array([
#                            (0.0, 0.0, 0.0),             # varful nasului
#                            (3, -41, -88.0515),        # barbie
#                            (-27, 15, -91.90),     # coltul stang al ochiului
#                            (25, 17, -91.23),      # coltul drept al ochiului
#                            (-13, -20, -88.9185),    # coltul stang al gurii
#                            (16, -19, -81.5105)      # coltul drept al gurii
#                        
#                        ])

            dimensiune = cadru.shape
            distanta_focala = dimensiune[1]
            center = (dimensiune[1]/2, dimensiune[0]/2)
            matricea_camerei = np.array(
                                     [[distanta_focala, 0, center[0]],
                                     [0, distanta_focala, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )
            
            #print ("Matricea camerei :\n {0}".format(matricea_camerei));
            
            coef_dist = np.zeros((4,1)) # Presupunem ca nu avem distorsiuni ale camerei
            (success, vector_rotatie, vector_translatie) = cv2.solvePnP(puncte_3D, puncte_2D, matricea_camerei, coef_dist, flags=cv2.CV_ITERATIVE)
#             
#            
#            #print ("Vector rotatie:\n {0}".format(vector_rotatie))
#            #print ("Vector translatie:\n {0}".format(vector_translatie))
#            
#            
#            # Proiectam directia capului din fiecare cadru
#            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), vector_rotatie, vector_translatie, matricea_camerei, coef_dist)
#            
            for p in puncte_2D:
                cv2.circle(cadru, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
#            
#            
            p1 = ( int(puncte_2D[0][0]), int(puncte_2D[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            print (puncte_2D, nose_end_point2D)
            cv2.line(cadru, p1, p2, (255,0,0), 2)
#           
            
 
        cv2.imshow("cadru", cadru)
 
        #  Programul se opreste la apasarea tastei q
        if cv2.waitKey(1) == ord('q'):
            break;