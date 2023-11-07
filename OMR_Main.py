import cv2
import numpy as np
import utils

########################################################################################################################
path = "1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 1, 4]
webcamFeed = True
cameraNo = 0
########################################################################################################################

cap = cv2.VideoCapture(cameraNo)

# Read image from path
# img = cv2.imread(path)

while True:
    if webcamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
    # Different types of images (Pre-Processing)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5,), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        # Finding contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        # Find Rectangles
        rectCon = utils.rectCountour(contours)
        biggestContour = utils.getCornerPoints(rectCon[0])
        gradePoints = utils.getCornerPoints(rectCon[1])

        # print(biggestContour)

        # Draw contour to biggest and second-biggest rectangle
        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

            biggestContour = utils.reorder(biggestContour)
            gradePoints = utils.reorder(gradePoints)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWrapColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            # cv2.imshow("Grade", imgGradDisplay)

            # Apply threshold
            imgWrapGray = cv2.cvtColor(imgWrapColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWrapGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThresh)
            # cv2.imshow("Test",boxes[2])

            # Getting non-zero pixel values of each box
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1

                if countC == choices:
                    countR += 1
                    countC = 0

            # print(myPixelVal)

            # Finding index value of the marking
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                # print("arr", arr)
                max_index = np.argmax(arr)
                # print("max_index", max_index)
                myIndexVal = np.where(arr == arr[max_index])
                # print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])

            # print(myIndex)

            # Grading
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)

            # print(grading)
            score = (sum(grading) / questions) * 100  # Final grade
            print(score)

            # Display Answers
            imgResult = imgWrapColored.copy()
            imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)

            # only answer to correct positions
            imgRawDrawing = np.zeros_like(imgWrapColored)
            imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)

            # Answer to image perspective
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWrap = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 255), 3)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWrap, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradDisplay, 1, 0)

        # Image array for showing image in stack
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWrapColored, imgThresh],
                      [imgResult, imgRawDrawing, imgInvWrap, imgFinal])
    except:
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    labels = [["Original", "Gray", "Blur", "Canny"],
              ["Contours", "Biggest Contour", "Wrap", "Threshold"],
              ["Result", "Raw Drawing", "Inv wrap", "Final"]]

    imgStacked = utils.stackImages(imageArray, 0.5, labels)

    # Show image in window
    # cv2.imshow("OMR", imgStacked)
    cv2.imshow("Final result", imgFinal)
    cv2.imshow("Stacked", imgStacked)

    # waitKey(0) will pause your screen because it will wait infinitely for keyPress on your keyboard and will not
    # refresh the frame(cap.read()) using your WebCam. waitKey(1) will wait for keyPress for just 1 millisecond, and
    # it will continue to refresh and read frame from your webcam using cap.read(). More clearly, Use debugger in
    # your code.When using waitKey(0) in the while loop, the debugger never crosses this statement and does not
    # refresh the frame and hence the frame output seems stable.Does not move.
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Final Result.jpg", imgFinal)
        cv2.waitKey(300)
