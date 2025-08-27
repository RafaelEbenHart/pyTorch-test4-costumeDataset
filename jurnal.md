Pytorch domain libraries
a. TrochVison (image based)
b. TrochText (Text Based)
c. TrochAudio (Audio Based)
D. TrochRec (making a recommendation)

setiap domain libraries pytorch memiliki datasetnya sendiri untuk dijadikan bahan latihan,contohnya pada torchVision terdapat FashionMNIST

Standard image classification data format:
data/
    photo/ <- overall dataset folder
        train/ <- train data
            class_name1/ <- class name as folder name
                image01.jpeg
                image02.jpeg
                ...
            class_name2/
                images03.jpeg
                image04.jpeg
                ...
            .../
                ...
                ...
        test/ <- test data
            class_name1/
                image05.jpeg
                image06.jpeg
                ...
            class_name2/
                image07.jpeg
                image08.jpeg
                ...
            .../
                ...
                ...
