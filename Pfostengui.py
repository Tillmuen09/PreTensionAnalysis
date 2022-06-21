# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:25:39 2022

@author: tmuenker
"""

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate
import os
from PIL import Image,ImageTk, ImageDraw
import numpy as np
import random



#%%

def drawLine(pos,window,key,im_original):
    im=im_original.copy()
    pos=int(im.size[0]/100*pos)
    draw = ImageDraw.Draw(im)
    draw.line((pos, im.size[1], pos,0 ), fill=255,width=5)
    image = ImageTk.PhotoImage(im)
    window[key].update(data=image)
    
def getdistance(min,max):
    a=abs(max[1]-min[1])
    b=abs(max[0]-min[0])
    return(np.sqrt(a**2+b**2))

def find_min_and_max(image,linestart,subpixel):
    line=image[:,linestart]
    
    #increase resolution by 5
    
    
    xachse=np.linspace(0, len(line)-1,len(line))
    xfine=np.linspace(0, len(line)-1,subpixel*len(line))
    f1 = interpolate.interp1d(xachse, line,kind = 'linear')
    yfine=f1(xfine)
    line=yfine
    
    #capline
    line[line>np.max(line)*0.6]=np.max(line)
    
    #plt.figure("lineprofile")
    #plt.plot(xfine,line)
    
    smoothline= savgol_filter(line, 21*subpixel, 1)
    #plt.plot(xfine,smoothline,label="smooth")
    #plt.legend()
    
    steigung=np.gradient(smoothline)
    
    #plt.figure("steigung")
    #plt.plot(xfine,steigung,".", label="steigung")
    #plt.legend()
    
    #such minimum und maximum
    
    minposition=np.argmin(steigung)
    maxposition=np.argmax(steigung[minposition:])+minposition
    #plt.plot(xfine[minposition],steigung[minposition],"o")
    #plt.plot(xfine[maxposition],steigung[maxposition],"o")
    
    #fit Gauss maximum
    
    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    popt,pcov = curve_fit(Gauss, xfine, steigung, p0=[steigung[maxposition], xfine[maxposition], 5*subpixel])
    
    #plt.plot(xfine,Gauss(xfine, *popt),'g--')
    
    truemaxpos=popt[1]
    
    #fit Gauss minimum
    
    def Gauss(x, a, x0, sigma):
        return -a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    popt,pcov = curve_fit(Gauss, xfine, steigung, p0=[steigung[minposition], xfine[minposition], 5*subpixel])
    
    #plt.plot(xfine,Gauss(xfine, *popt),'g--')
    
    trueminpos=popt[1]
    
    return(trueminpos,truemaxpos)

def getDistance(image,start,subpixel,um_per_pixel,search_range=100):

   
    linestarts=np.linspace(start-search_range,start+search_range,2*search_range+1)
    all_min=[]
    all_max=[]
    for linestart in linestarts:
        linestart=int(linestart)
        min,max=find_min_and_max(image, linestart,subpixel)
        all_min.append([linestart,min])
        all_max.append([linestart,max])
     
    shortest=np.infty
    all_distances=[]
    global_min=np.infty
    global_max=np.infty
    for min in all_min:
        for max in all_max:
            distance=getdistance(min, max)
            all_distances.append(distance)
            if distance<shortest:
                shortest=distance
                global_min=min
                global_max=max
    print(shortest)
    # plt.figure()
    # plt.hist(all_distances,50)
    
    plt.figure()
    plt.imshow(image)
    plt.plot([global_min[0],global_max[0]],[global_min[1],global_max[1]],'r-')
    plt.show()
    
    distance_in_um=shortest*um_per_pixel
    
    print("Minimum distance is "+str(distance_in_um)+"um")
    return(distance_in_um)

#%%


theme=random.choice(sg.theme_list())

sg.theme(theme)


top_row=[
    [sg.Text('Search range [px]:'),
     sg.InputText(default_text='100',key='search_range'),
     sg.Text('um per pixel:'),
     sg.Input(default_text='2.5',key='um per pixel'),
     sg.Text('Spring constant [uN / um]'),
     sg.Input(default_text='40',key='spring_constant')
     ]
    ]

left_column=[
    [sg.T("")], [sg.Text("Choose day 1 image: "), sg.InputText(key='image1',enable_events=True), sg.FileBrowse()],
    [sg.Image(size=(300,300),key='-IMAGE1-')],
    [sg.Slider(range=(1, 100), orientation='h', size=(40, 20), default_value=50,key='slider1',enable_events=True)]
    
    ]

right_column=[[sg.T("")], [sg.Text("Choose final day image: "), sg.Input(key='image2',enable_events=True), sg.FileBrowse()],
              [sg.Image(size=(300,300),key='-IMAGE2-')],
              [sg.Slider(range=(1, 100), orientation='h', size=(40, 20), default_value=50,key='slider2',enable_events=True)]
              ]

bottom_row=[
    [sg.Button('Analyse',size=(30,3),enable_events=True,key='analyse')]
            ]
result_row=[
    [sg.Text('Deflection: '),sg.Text('analyze first',size=(30,1),key='result')],
    [sg.Text('Force:'),sg.Text('',size=(30,1),key='force')]
    ]


layout=[
        [top_row],
        [sg.Column(left_column),sg.VSeparator(),sg.Column(right_column)],
        [bottom_row],
        [result_row],
        
        ]



window=sg.Window(title='Pfosten App',layout=layout,finalize=True,element_justification='center')

start_1=50
start_2=50

subpixel=5

while True:
    
    plt.ioff()
    event,values=window.read()
    try:
        search_range=int(values['search_range'])
        um_per_pixel=float(values['um per pixel'])
        spring_constant=float(values['spring_constant'])
    except:
        print('Could not read numbers')

    
    
    if event=='OK' or event == sg.WINDOW_CLOSED:
        break
    
    #When loaded Plot first image
    if event == 'image1':
        path1=values['image1']
        try:
            im1=Image.open(path1)
            im1.thumbnail((400, 400))
            photo_img1 = ImageTk.PhotoImage(im1)
            window['-IMAGE1-'].update(data=photo_img1)
            drawLine(start_1,window,'-IMAGE1-',im1)
        except:
            print('Cant open file')
            
            
    #When loaded plto second image        
    elif event == 'image2':
        path2=values['image2']
        try:
            im2=Image.open(path2)
            im2.thumbnail((400, 400))
            photo_img2 = ImageTk.PhotoImage(im2)
            window['-IMAGE2-'].update(data=photo_img2)
            drawLine(start_2,window,'-IMAGE2-',im2)
            
        except:
            print('Cant open file')
            
    #When slider changes, draw image
    elif event== 'slider1':
        try:
            start_1=values['slider1']
            drawLine(start_1,window,'-IMAGE1-',im1)
        except:
            print('Cant update im 1')
    elif event =='slider2':
        try:
            start_2=values['slider2']
            drawLine(start_2,window,'-IMAGE2-',im2)
        except:
            print('Cant update im 2')
            
    if event == 'analyse':
        
        try:
            #Check if images are loaded
            try:
                image1=Image.open(path1)
                image1=np.array(image1)
            except:
                print('Cant find im 1')
                pass
            try:
                image2=Image.open(path2)
                image2=np.array(image2)
            except:
                print('Cant find image 2')
                pass
            
            plt.ion()
            start1=int(np.shape(image1)[1]/100*start_1)
            start2=int(np.shape(image2)[1]/100*start_2)
            distance1=getDistance(image1, start1, subpixel,um_per_pixel,search_range)
            distance2=getDistance(image2, start2, subpixel,um_per_pixel,search_range)
            
            final_distance=abs(distance2-distance1)
            result_string=str(final_distance)+' um'
            
            force_string=str(final_distance*spring_constant*1e-3)+' mN'
            window['result'].update(result_string)
            window['force'].update(force_string)
            
        except:
            print('some thing went wrong analysing')
        
        
        
        
        

        
            
    
    
window.close()