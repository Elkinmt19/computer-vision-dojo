# Built-in imports
import os
import csv

# External imports
import cv2 as cv
import glob
import numpy as np
import shutil
import wx

# Important constants variables
PERCENT = 1
WIDTH = 920
HEIGHT = 640
pathImg = ''
path = ''
fileCSV = 'train.csv'
jumps = 0
countVideos = 1
countFrames = 1
nameImg = 'img'
label = 'person'
labelOK = False
W = 0
H = 0
xmin = 0
ymin = 0
bandClick = False
bandClass = False
bandRect = False
CLASSES = ["person", "mannequin", "dog", "plant", "table", "chair", "tree", "washbasin","bathroom","stairs","laptop","container"]

cv.namedWindow('image')
img = img = np.zeros((240, 480, 3), np.uint8)
cv.imshow('image',img)

class textBoxLabel(wx.Dialog):
    def __init__(self, parent, id, title):
        global label, labelOK
        
        wx.Dialog.__init__(self, parent, id, title, size = (500, 200))
        self.Center()

        self.listClass = ["person", "mannequin", "dog", "plant", "table", "chair", "tree", "washbasin","bathroom","stairs","laptop","container"]
        self.rbClass = wx.RadioBox(self, pos=(10, 0), choices = self.listClass, majorDimension=4, style = wx.RA_SPECIFY_COLS)
        self.rbClass.SetSelection(self.listClass.index(label))
        #self.txtB = wx.TextCtrl(self, value = label, pos = (10, 10), size = (40, -1))
        self.btnOk = wx.Button(self, 2, "OK", (60, 45), (40, -1))
        self.btnOk.SetFocus()
        self.Bind(wx.EVT_BUTTON, self.OnOk)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
    def OnClose(self, event):
        self.Destroy()
    def OnOk(self, event):
        global label, labelOK
        label = self.listClass[self.rbClass.GetSelection()]
        #label = self.txtB.GetValue()
        labelOK = True
        self.Close(True)
        self.Destroy()

def readCSV():    
    with open('train.csv') as File:
        reader = csv.reader(File)
        for row in reader:
            file = 'imagesTrain\\' + row[0]
            print (file)
            if row[0] != 'filename':
                if os.path.isfile(file):
                    img = cv.imread(file)
                    cv.rectangle(img, (int(row[4]), int(row[5])), (int(row[6]), int(row[7])), (255, 0, 0), 2)
                    cv.imshow("image", img)
##                    shutil.move(row[0], row[0][:43] + "imgTs" + row[0][53:])
                if cv.waitKey() & 0xFF == ord('q'):
                    break
        cv.destroyAllWindows()
##readCSV()

def writeCSV(data, file):
    myFile = open(file, 'a', newline = '')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(data)
    print("Writing complete ", data[3])
    
def nothing(x):
    pass

def click_mouse(event, x, y, flags, param):
    global bandClick, xmin, ymin, pathImg, bandClass, img, tagged, bandRect, imgRect, label, labelOK, W, H
    h, w = img.shape[:2]
    if not bandClick:
        if bandRect:
            imgLine = imgRect.copy()
        else:
            imgLine = img.copy()
        cv.line(imgLine, (0, y), (w, y), (128, 128, 128), 2)
        cv.line(imgLine, (x, 0), (x, h), (128, 128, 128), 2)
        cv.imshow('image',imgLine)
    if event == cv.EVENT_LBUTTONDOWN and not bandClass:
        xmin = x
        ymin = y
        bandClick = True
    if bandClick:
        imgRect = img.copy()
        cv.rectangle(imgRect, (xmin, ymin), (x, y), (255, 0, 0), 2)    
        cv.imshow('image',imgRect)
        bandRect = True
    if event == cv.EVENT_LBUTTONUP and not bandClass:
        bandClick = False
        bandClass = True

        dialog2 = textBoxLabel(None, -1, "label")
        dialog2.ShowModal()
        #["person", "mannequin", "dog", "plant", "table", "chair", "tree", "washbasin","bathroom","stairs","laptop","container"]
        if labelOK and (label == 'person' or label == 'mannequin' or label == 'dog' or label == 'plant' or label == 'table' or label == 'chair' or label == 'tree' or label == 'washbasin' or label == 'bathroom' or label == 'stairs' or label == 'laptop' or label == 'container'):
            labelOK = False
            tagged = True
            xmax = x
            ymax = y
            if xmin > x:
                xmax = xmin
                xmin = x
            if ymin > y:
                ymax = ymin
                ymin = y
            if xmax > w:
                xmax = w
            if ymax > h:
                ymax = h
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            divH = (H*PERCENT) / h
            divW = (W*PERCENT) / w
            writeCSV([pathImg, int(W*PERCENT), int(H*PERCENT), label, int(xmin * divW), int(ymin * divH),
                     int(xmax * divW), int(ymax * divH)], fileCSV)
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv.putText(img, label, (xmin, ymin-1), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow('image',img)
        bandClass = False
        bandRect = False
        
    return 0

def predict(img, fileTxt):
    global PERCENT, label, labelOK, tagged, imgRect, bandRect, bandClass, pathImg, W, H
    h, w = img.shape[:2]
    with open(fileTxt) as File:
        reader = csv.reader(File, delimiter=' ')
        for row in reader:
            idx = int(row[0])
            label = CLASSES[idx]
            cx = int(float(row[1]) * w)
            cy = int(float(row[2]) * h)
            startX = cx - int(float(row[3]) * w / 2)
            endX = cx + int(float(row[3]) * w / 2)
            startY = cy - int(float(row[4]) * h / 2)
            endY = cy + int(float(row[4]) * h / 2)
 
            if endX > w:
                endX = w
            if endY > h:
                endY = h
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0
            imgRect = img.copy()
            cv.rectangle(imgRect, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv.imshow('image',imgRect)
            bandRect = True
            bandClass = True
            
            dialog2 = textBoxLabel(None, -1, "label")
            dialog2.ShowModal()

            if labelOK and (label == 'person' or label == 'mannequin' or label == 'dog' or label == 'plant' or label == 'table' or label == 'chair' or label == 'tree' or label == 'washbasin' or label == 'bathroom' or label == 'stairs' or label == 'laptop' or label == 'container'):
                labelOK = False
                tagged = True
                divH = (H*PERCENT) / h
                divW = (W*PERCENT) / w
                writeCSV([pathImg, int(W*PERCENT), int(H*PERCENT), label, int(startX * divW), int(startY * divH),
                          int(endX * divW), int(endY * divH)], fileCSV)
                cv.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv.putText(img, label, (startX, startY-1), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.imshow('image',img)

            bandRect = False
            bandClass = False
            
def folder(path, predictOK = False):
    global image, pathImg, img, tagged, jumps, countFrames, W, H
    i = 0 - countFrames
    countFrames = 0
    for p in glob.glob(path + "\*.jpg"):
        countFrames = countFrames + 1
        print (p)
        i = i + 1
        if i == jumps+1 or i == 0:
            i = 0
            image = cv.imread(p)
            H, W = image.shape[:2]
            img = cv.resize(image, (WIDTH, HEIGHT))
            imgR = cv.resize(image, (640, 480))
            cv.imshow('image',img)
            tagged = False

            p2 = os.getcwd() + "\\imagesTrain" + p[len(path):]
            p3 = os.getcwd() + "\\imagesResize" + p[len(path):]
            pathImg = p[len(path)+1:]
            
            if predictOK:
                fileTxt = p[:p.find('.')] + '.txt'
                predict(img, fileTxt)

            key = ''
            while True:
                key = cv.waitKey()
                if key == ord('q') or key == ord(' '):
                    break
            if key == ord('q'):
                break
            if tagged:
                shutil.copy(p, p2)                
                cv.imwrite(p3, imgR)
            f = open("countFrames.txt", 'w+')
            f.write('0\n')
            f.write(str(countFrames))
            f.close()

def video(path, predictOK = False):
    global nameImg, pathImg, img, tagged, jumps, countVideos, countFrames, W, H, PERCENT
    stop = False
    j = 0
    a = 0
    for p in glob.glob(path + "\*.mp4"):
        j = j + 1
        print (p)
        if j > countVideos - 1:
            i = 0 - countFrames
            countFrames = 0
            cap = cv.VideoCapture(p)
            k = 0
            while cap.isOpened():        
                ret, image = cap.read()                
                if not ret:
                    break
                k = k + 1
                print ("New frame")                
                i = i + 1
                if i == jumps+1 or i == 0:
                    i = 0
                    H, W = image.shape[:2]
                    img = cv.resize(image, (WIDTH, HEIGHT))
                    cv.imshow('image',img)
                    tagged = False

                    while True:                
                        p2 = os.getcwd() + "\\imagesTrain\\" + nameImg + str(a) + ".jpg"
                        if os.path.isfile(p2):
                            a = a + 1
                        else:
                            break
                    pathImg = nameImg + str(a) + ".jpg"

                    #if predictOK:
                        #predict(img)
                        
                    key = ''
                    while True:
                        key = cv.waitKey()
                        if key == ord('q') or key == ord(' '):
                            break
                    if key == ord('q'):
                        stop = True
                        break
                    if tagged:                       
                        imgR = cv.resize(image, (0,0), fx=PERCENT, fy=PERCENT)
                        cv.imwrite(p2, imgR)
                        a = a + 1
                f = open("countFrames.txt", 'w+')
                f.write(str(j) + '\n')
                f.write(str(k))
                f.close()
            countVideos = j
            cap.release()
            if stop:
                countFrames = k
                break
                    
class config(wx.Dialog):
    def __init__(self, parent, id, title):
        global path, fileCSV, jumps, file_pb, file_pbtxt, countVideos, countFrames, nameImg
        self.mode = 0
        self.capture = 0
        wx.Dialog.__init__(self, parent, id, title, size = (440, 290))
        self.Center()
        
        listMode = ['Manual', 'Semi-Auto']
        self.rbMode = wx.RadioBox(self, pos=(20, 10), choices = listMode, majorDimension=2, style = wx.RA_SPECIFY_COLS)
        listCapture = ['Video', 'Imagenes']
        self.rbCapture = wx.RadioBox(self, pos=(220, 10), choices = listCapture, majorDimension=2, style = wx.RA_SPECIFY_COLS)
        wx.StaticText(self, -1, 'Ruta :', (20, 60))
        self.TxtPath = wx.TextCtrl(self, value = path, pos = (120, 60), size = (280, 20))
        wx.StaticText(self, -1, 'Archivo CSV:', (20, 90))
        self.TxtFileCSV = wx.TextCtrl(self, value = fileCSV, pos = (120, 90), size = (280, 20))
        wx.StaticText(self, -1, 'Saltos de frames :', (20, 120))
        self.TxtJumps = wx.TextCtrl(self, value = str(jumps), pos = (120, 120), size = (80, 20))
        wx.StaticText(self, -1, 'Videos :', (215, 120))
        self.TxtCount = wx.TextCtrl(self, value = str(countVideos), pos = (285, 120), size = (35, 20))
        wx.StaticText(self, -1, 'Fr :', (330, 120))
        self.TxtCountFr = wx.TextCtrl(self, value = str(countFrames), pos = (350, 120), size = (50, 20))        
        wx.StaticText(self, -1, 'Nombre Img :', (20, 150))
        self.Txt_name = wx.TextCtrl(self, value = nameImg, pos = (120, 150), size = (280, 20))
        btnCancel = wx.Button(self, 1, "Cancelar", (110, 190), (100, -1))
        btnOk = wx.Button(self, 2, "Aceptar", (220, 190), (100, -1))

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, btnCancel)
        self.Bind(wx.EVT_BUTTON, self.OnOk, btnOk)

    def OnClose(self, event):
        self.Destroy()
    def OnCancel(self, event):
        self.Close(True)
        self.Destroy()
    def OnOk(self, event):
        global path, fileCSV, jumps, countVideos, countFrames, file_pb, file_pbtxt, net, nameImg

        self.mode = self.rbMode.GetSelection()
        self.capture = self.rbCapture.GetSelection()
        path = self.TxtPath.GetValue()
        fileCSV = self.TxtFileCSV.GetValue()
        jumps = int(self.TxtJumps.GetValue())
        countVideos = int(self.TxtCount.GetValue())
        countFrames = int(self.TxtCountFr.GetValue())
        nameImg = self.Txt_name.GetValue()
        self.Show(False)

        f = open("config.txt", 'w+')
        f.write(path + '\n')
        f.write(fileCSV + '\n')
        f.write(str(jumps) + '\n')
        f.write(nameImg + '\n')
        f.close()           
        cv.setMouseCallback("image", click_mouse)
        if not os.path.isfile(fileCSV):
            writeCSV(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"], fileCSV)
        if not os.path.exists("imagesTrain"):
            os.mkdir("imagesTrain")

        if self.mode == 0:
            if self.capture == 0:
                video(path)
            elif self.capture == 1:
                folder(path)
        elif self.mode == 1:
            if self.capture == 0:
                video(path, True)
            elif self.capture == 1:
                folder(path, True)

        self.Show(True)
        self.TxtCount.SetValue(str(countVideos))
        self.TxtCountFr.SetValue(str(countFrames))

if os.path.exists("config.txt"):      
    f = open("config.txt", 'r+')
    line = f.readline()
    if len(line) > 1:
        path = line[:line.find('\n')]
    line = f.readline()
    if len(line) > 1:
        fileCSV = line[:line.find('\n')]
    line = f.readline()
    if len(line) > 0:
        jumps = int(line[:line.find('\n')])
    line = f.readline()
    if len(line) > 1:
        nameImg = line[:line.find('\n')]
    f.close()
if os.path.exists("countFrames.txt"):      
    f = open("countFrames.txt", 'r+')
    line = f.readline()
    if len(line) > 0:
        countVideos = int(line)
    line = f.readline()
    if len(line) > 0:
        countFrames = int(line)
    f.close()
MyApp = wx.App(False)
dialog = config(None, -1, "Config")
dialog.Show(True)
MyApp.MainLoop()
cv.destroyAllWindows()
