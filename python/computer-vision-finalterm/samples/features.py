import cv2
import numpy as np
import xlsxwriter
import glob
vectorFolders = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
row = 0
col = 0
ii = 1
workbook = xlsxwriter.Workbook('DataNumberAll.xlsx')
worksheet = workbook.add_worksheet('carac')
path='caracteres_motos/'
for jj in range(0,len(vectorFolders)):
    for path in glob.glob(path+vectorFolders[jj]+"/*.jpg"):
        img = cv2.imread(path,0)
        imgColor = cv2.imread(path,1)
        imgColor = cv2.resize(imgColor, (20,35))
        img = cv2.resize(img, (20,35))
        ret, imgBin = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnt,hie = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.imshow("img",imgBin)
        #cv2.waitKey(1)
        Vectorcaract = []
        for cont in cnt:
            x,y,w,h = cv2.boundingRect(cont)            
            #print(cv2.contourArea(cont))
            if(cv2.contourArea(cont)>100):
                #cv2.rectangle(imgColor,(x,y),(x+w,y+h),(0,0,255),2)
                #cv2.imshow("imgColor",imgColor)
                for j in range(0,4):  #Se dividió 20 y 40 por 5
                    for i in range(0,6):
                        imgRoi_2 = img[i*5:(i+1)*5, j*5:(j+1)*5] #Acá se estan haciento recortes de 5x5 a la imagen, como filtros
                        hh,ww = imgRoi_2.shape[:2]
                        #print(h,w)
                        #cv2.imshow("imgRoi_2",imgRoi_2)
                        #cv2.waitKey(0)
                        #print("in...")
                        for k in range(0,hh):
                            #print("por filas...")
                            #cv2.imshow("imgRoi_2_filas",cv2.resize(imgRoi_2[k:k+1,:],(50,10))) #Esto es para verlo +grande
                            valRow = cv2.countNonZero(imgRoi_2[k:k+1,:])   # Contar pixeles blancos                         
                            diag = np.diagonal(imgRoi_2[k:k+1,:])                            
                            val_diag = (diag[0])/255.0
                            valRow = valRow/hh
                            #print("valRow", valRow)
                            #print("val_diag", val_diag)
                            Vectorcaract.append(valRow)
                            Vectorcaract.append(val_diag)
                            #cv2.waitKey(0)
                        for m in range(0,ww):
                            #print("por columnas...")
                            #cv2.imshow("imgRoi_2_columnas",cv2.resize(imgRoi_2[:,m:m+1],(10,50)))
                            valCol = cv2.countNonZero(imgRoi_2[:,m:m+1])
                            diag = np.diagonal(imgRoi_2[:,m:m+1])
                            val_diag = (diag[0])/255.0
                            valCol = valCol/hh
                            #print("valCol", valCol)
                            #print("val_diag", val_diag)
                            Vectorcaract.append(valCol)
                            Vectorcaract.append(val_diag)
                            #cv2.waitKey(0)
                #print((Vectorcaract))
                #cv2.waitKey(0)
                for carac in (Vectorcaract):
                    worksheet.write(row, col, vectorFolders[jj])
                    worksheet.write(row, ii, carac)
                    ii=ii+1
                ii=1
                row += 1
workbook.close()
cv2.destroyAllWindows()