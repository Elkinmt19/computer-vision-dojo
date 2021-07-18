# Others imports 
import cv2 as cv

def load_image(path, mode):
    return (cv.imread(path, mode))

def show_image(nameWin, img):
    cv.imshow(nameWin, img)

def closeWin():
    cv.destroyAllWindows()

def get_pixel_image(img):
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            print(img[i,j])

def main():
    colorImage = load_image("python/assets/imgs/Tony_Stark.jpeg", 1)
    show_image("Tony Stark", colorImage)
    get_pixel_image(colorImage)
    cv.waitKey(0)
    colorImage()


if __name__ == '__main__':
    main()