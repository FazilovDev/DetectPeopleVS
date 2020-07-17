from tkinter import *
from tkinter import filedialog as fd
import cv2 as cv
from CrowdNet import CrowdNetModel
from PIL import Image, ImageFile,ImageTk
from IPython.display import display
import numpy as np
import threading
import time
import numpy

class VideoSystem(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")

        self.parent = parent
        self.width = self.winfo_screenwidth()  # Ширина экрана
        self.height = self.winfo_screenheight()  # Высота экрана

        self.parent.title("DetectPeopleVS")
        self.pack(fill=BOTH, expand=True)

        self.fullscreen = False  # Переменная для отслеживания развертывания экрана
        self.is_mask = BooleanVar()  # Сегментация людей по маске
        self.is_box = BooleanVar()  # Сегментация людей по прямоугольникам
        self.is_point = BooleanVar()  # Сегментация людей по точкам
        self.is_mask.set(True)
        self.is_box.set(False)
        self.is_point.set(False)

        self.cap = None  # Переменная для работы с вебкамерой или готовым видео
        self.ch_scale_frame = 0  # Переменная для выбора количества обрабатываемых кадров
        self.filename = ""  # Название видеофайла

        self.model = CrowdNetModel()  # Задаем модель нейронной сети

        self.main_menu_widgets = []  # Контейнер для виджетов главного меню
        self.info_menu_widgets = []  # Контейнер для виджетов информационного меню
        self.ready_video_widgets = []  # Контейнер для виджетов меню обработки готового видео
        self.ready_image_widgets = []  # Контейнер для виджетов меню обработки изображения

        self.create_main_menu()  # Создаем главное меню
        

    def create_main_menu(self):
        '''
        Метод создания виджетов главного меню
        '''
        frame1 = Frame(self)
        frame1.pack(fill=X)
        frame1.config(bg="white")
        self.main_menu_widgets.append(frame1)
        text_info_menu = Label(frame1, text="Выберите режим работы:", font=("Arial Bold", 20))
        text_info_menu.config(bg="white")
        text_info_menu.pack()
        btn_switch_fullscreen = Button(frame1, text="Оконный/Полноэкранный", command=self.switch_window)
        btn_switch_fullscreen.pack(side=RIGHT, expand=True)
        
        frame2 = Frame(self)
        frame2.pack(fill=X)
        frame2.config(bg="white")
        delta_y = self.width * 0.005
        self.main_menu_widgets.append(frame2)
        btn_about_app = Button(frame2, text="Справка", width=int(self.width*0.02), height=int(self.height*0.002), command=self.goto_info_menu)
        btn_about_app.pack(pady=delta_y)


        btn_ready_video = Button(frame2, text="Работа по записанному видео", width=int(self.width*0.02), height=int(self.height*0.002),command=self.choice_videofile)
        btn_ready_video.pack(pady=delta_y)

        btn_real_time_video = Button(frame2, text="Работа в настоящем времени(веб-камера)",width=int(self.width*0.02), height=int(self.height*0.002) )
        btn_real_time_video.pack(pady=delta_y)

        btn_ready_image = Button(frame2, text="Обработать фото",width=int(self.width*0.02), height=int(self.height*0.002), command=self.choice_imagefile )
        btn_ready_image.pack(pady=delta_y)

        frame3 = Frame(self)
        frame3.pack(fill=BOTH)
        frame3.config(bg="white")
        self.main_menu_widgets.append(frame3)
        text_segment = Label(frame3, text="Конфигурация", font=("Arial Bold", 12))
        text_segment.pack(pady=delta_y*2, expand=True)
        text_segment.config(bg="white")

        state_webcam = 'не найдена'
        text_info_webcam = Label(frame3, text="Состояние вебкамеры: "+state_webcam, font=("Arial Bold", 12))
        text_info_webcam.pack(pady=delta_y, expand=True)
        text_info_webcam.config(bg="white")
        if (cv.VideoCapture(0).isOpened()):
            state_webcam = 'найдена'
        else:
            btn_real_time_video.configure(state=DISABLED)

        chk_mask = Checkbutton(frame3, text="Маска", var=self.is_mask)
        chk_mask.pack(pady=delta_y,expand=True)
        chk_mask.config(bg="white")

        chk_box = Checkbutton(frame3, text="Рамка", var=self.is_box)
        chk_box.pack(pady=delta_y,expand=True)
        chk_box.config(bg="white")

        chk_point = Checkbutton(frame3, text="Точка", var=self.is_point)
        chk_point.pack(pady=delta_y,expand=True) 
        chk_point.config(bg="white")

    def switch_window(self):
        '''
        Метод перевода приложения в полноэкранный/оконный режим
        '''
        if (self.fullscreen):
            self.parent.overrideredirect(0)  # Метод у окна, позволяющий сменить полноэкранный/оконный режим
            self.fullscreen = False
        else:
            self.fullscreen = True
            self.parent.overrideredirect(1)
    
    def goto_info_menu(self):
        '''
        Метод перехода из главного меню в информационное
        '''
        self.clear_widgets(self.main_menu_widgets)
        self.create_info_menu()
    
    def clear_widgets(self, widgets):
        '''
        Метод очищения экрана от виджетов
        '''
        for item in widgets:
            item.destroy()
    
    def choice_imagefile(self):
        '''
        Метод для выбора файла изображения и загрузки меню обработки
        '''
        self.filename = fd.askopenfilename(defaultextension='.jpg', filetypes=[('JPG', '.jpg'),('PNG', '.png')])
        self.clear_widgets(self.main_menu_widgets)
        self.create_menu_processing_image()
    
    def goto_main_menu_from_image(self):
        '''
        Метод перехода в главное меню из меню обработки изображения
        '''
        self.clear_widgets(self.ready_image_widgets)
        self.create_main_menu()
    
    def create_menu_processing_image(self):
        '''
        Метод отрисовки меню обработки изображения
        '''
        frame1 = Frame(self)
        frame1.pack()
        self.ready_image_widgets.append(frame1)
        btn_exit = Button(frame1, text="Выйти в меню", command=self.goto_main_menu_from_image)
        btn_exit.pack()
        self.lbl_count_people = Label(frame1, font=("Arial Bold", 16))
        self.lbl_count_people.pack()
        frame2 = Frame(self)
        frame2.pack()
        self.ready_image_widgets.append(frame2)

        self.frame_video = Label(frame2)
        self.frame_video.pack(side=LEFT)
        self.frame_mask = Label(frame2)
        self.frame_mask.pack(side=LEFT)

        frame3 = Frame(self)
        frame3.pack()
        self.ready_image_widgets.append(frame3)

        self.frame_box = Label(frame3)
        self.frame_box.pack(side=LEFT)
        self.frame_point = Label(frame3)
        self.frame_point.pack(side=LEFT)

        img = Image.open(str(self.filename))
        image = cv.cvtColor(numpy.array(img), cv.COLOR_RGB2BGR)
        count_people = self.model.predict(image)

        self.lbl_count_people['text'] = "Количество людей: " + str(count_people)
        fr = self.get_image_from_frame(image)
        self.frame_video.image = fr
        self.frame_video.configure(image=fr)

        if self.is_mask.get():
            f = self.model.magix(mask=self.is_mask.get())
            f1 = self.get_image_from_frame(f)
            self.frame_mask.image = f1
            self.frame_mask.configure(image=f1)
        if self.is_box.get():
            f = self.model.magix(box=self.is_box.get())
            f2 = self.get_image_from_frame(f)
            self.frame_box.image = f2
            self.frame_box.configure(image=f2)
        if self.is_point.get():
            f = self.model.magix(point=self.is_point.get())
            f3 = self.get_image_from_frame(f)
            self.frame_point.image = f3
            self.frame_point.configure(image=f3)

    def create_info_menu(self):
        '''
        Метод создания виджетов информационного меню
        '''
        frame = Frame(self)
        frame.pack()
        self.info_menu_widgets.append(frame)
        btn_exit = Button(frame, text="Назад", width=int(self.width*0.02), height=int(self.height*0.002),command=self.goto_main_menu_from_info)
        btn_exit.pack()
        info = "Справка:\nПриложение работает в 3х режимах:\n1.В реальном времени с помощью вебкамеры\n2.По готовому видео\n3.По готовому изображению\n\
            \nОбученная нейронная сеть позволяет проводить сегментацию людей на изображении\nЕсть сегментация маской, прямоугольником и точкой\
            \nВ зависимости от технических характеристик компьютера можно поставить количество обрабатываемых кадров.\
            \nНа слабых компьютерах для проверки качества работы лучше ставить меньше количество кадров(особенно, если нет CUDA)"
        text = Label(frame, text=info,font=("Arial Bold", 16))
        text.pack(fill=BOTH, expand=True)

    def goto_main_menu_from_info(self):
        '''
        Метод перехода в главное меню из информационного
        '''
        self.clear_widgets(self.info_menu_widgets)
        self.create_main_menu()
    
    def goto_main_menu_from_ready(self):
        '''
        Метод перехода в главное меню из меню обработки готового видео
        '''
        self.clear_widgets(self.ready_video_widgets)
        self.is_work_thread = False
        self.create_main_menu()
    
    def choice_videofile(self):
        '''
        Метод для выбора видеофайла
        '''
        self.filename = fd.askopenfilename(defaultextension='.avi', filetypes=[('AVI video', '.avi'),('MP4 video', '.mp4')])
        self.clear_widgets(self.main_menu_widgets)
        self.create_menu_ready_video()
    
    def on_scale_frames(self, val):
        '''
        Метод для работы с ползунком
        '''
        v = int(float(val))
        self.ch_scale_frame.set(v)
    
    def create_menu_ready_video(self):
        '''
        Метод создания виджетов меню обработки готового видео
        '''
        if self.filename == "":
            self.create_main_menu()
            return
        frame1 = Frame(self)
        frame1.pack()
        self.ready_video_widgets.append(frame1)
        btn_exit = Button(frame1, text="Выйти в меню", command=self.goto_main_menu_from_ready)
        btn_exit.pack()

        self.cap = cv.VideoCapture(self.filename)
        count_frame = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

        s = "Количество кадров в видео: " + str(count_frame)
        count_frame_lbl = Label(frame1, text=s, font=("Arial Bold", 16))
        count_frame_lbl.pack()

        scale_frame = Scale(frame1, length=600, from_=0, to=count_frame, orient=HORIZONTAL,tickinterval=100, resolution=24, command=self.on_scale_frames)
        scale_frame.pack()
        
        self.ch_scale_frame = IntVar()
        scale_frame_lbl = Label(frame1, text=0, textvariable=self.ch_scale_frame)
        scale_frame_lbl.pack()

        choice_frame_btn = Button(frame1, text="Обработать", command=self.processing_menu_ready_video)
        choice_frame_btn.pack()
    
    def get_image_from_frame(self, frame):
        '''
        Метод для конвертирования из тензора в изображение
        '''
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk
    
    def processing_thread(self):
        '''
        Метод для обработки видео в отдельном потоке
        '''
        type_file = self.filename[-4:]
        if self.is_mask.get():
            name = self.filename.replace(type_file, '_mask' + type_file)
            mask_video = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'),24,(640,480))
        if self.is_box.get():
            name = self.filename.replace(type_file, '_box' + type_file)
            box_video = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'),24,(640,480))
        if self.is_point.get():
            name = self.filename.replace(type_file, '_point' + type_file)
            point_video = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'),24,(640,480))

        i =  0
        while i < self.ch_scale_frame.get() and self.is_work_thread:
            ret, frame = self.cap.read()
            if not ret:
                break

            fr = self.get_image_from_frame(frame)
            self.frame_video.image = fr
            self.frame_video.configure(image=fr)

            count_people = self.model.predict(frame)
            f1 = f2 = f3 = None
            if self.is_mask.get():
                fr = self.model.magix(mask=self.is_mask.get())
                f1 = self.get_image_from_frame(fr)
                mask_video.write(fr)

            if self.is_box.get():
                fr = self.model.magix(box=self.is_box.get())
                f2 = self.get_image_from_frame(fr)
                box_video.write(fr)

            if self.is_point.get():
                fr = self.model.magix(point=self.is_point.get())
                f3 = self.get_image_from_frame(fr)
                point_video.write(fr)
            
            self.lbl_count_people['text'] = "Количество людей: " + str(count_people)
            fr = self.get_image_from_frame(frame)
            self.frame_video.image = fr
            self.frame_video.configure(image=fr)
            if f1 is not None:
                self.frame_mask.image = f1
                self.frame_mask.configure(image=f1)
            if f2 is not None:
                self.frame_box.image = f2
                self.frame_box.configure(image=f2)
            if f3 is not None:
                self.frame_point.image = f3
                self.frame_point.configure(image=f3)

            time.sleep(0.1)
            
            i+=1

        if self.is_mask.get():
            mask_video.release()
        if self.is_box.get():
            box_video.release()
        if self.is_point.get():
            point_video.release()
        self.cap.release()


    def processing_menu_ready_video(self):
        '''
        Метод для отрисовки меню обработки видео
        '''
        self.clear_widgets(self.ready_video_widgets)
        frame1 = Frame(self)
        frame1.pack()
        self.ready_video_widgets.append(frame1)

        btn_exit = Button(frame1, text="Выйти в меню", command=self.goto_main_menu_from_ready)
        btn_exit.pack()

        self.lbl_count_people = Label(frame1, font=("Arial Bold", 16))
        self.lbl_count_people.pack()
        frame2 = Frame(self)
        frame2.pack()
        self.ready_video_widgets.append(frame2)

        self.frame_video = Label(frame2)
        self.frame_video.pack(side=LEFT)
        self.frame_mask = Label(frame2)
        self.frame_mask.pack(side=LEFT)

        frame3 = Frame(self)
        frame3.pack()
        self.ready_video_widgets.append(frame3)

        self.frame_box = Label(frame3)
        self.frame_box.pack(side=LEFT)
        self.frame_point = Label(frame3)
        self.frame_point.pack(side=LEFT)
        self.is_work_thread = True
        self.thread = threading.Thread(target=self.processing_thread)
        self.thread.start()
        
def main():
    root = Tk()
    root.geometry("{}x{}".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    app = VideoSystem(root)
    root.mainloop()

if __name__ == '__main__':
    main()

