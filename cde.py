
from tkinter import *
from PIL import ImageTk, Image
import tkinter.font as TkFont
from tkinter.filedialog import askopenfilename
import torch
from torchvision import transforms, models
root = Tk()
root.geometry("450x400")
root.title("Login Form")
model_path = 'C:\\Users\\sivas\\Downloads\\logo_classification_model_epoch500.pth'
test_image_path = ':\\Users\\sivas\\Downloads\\archive_logo_fake\\Output\\Zara\\000001.jpg'
# Load the model
model = models.resnet50(pretrained=False)
num_classes = 2  
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)

image_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
])

       
def open_win():

   def clickMe():
    global test_image_path
    file_name = askopenfilename()
    print(file_name)
    test_image_path=file_name
    print(test_image_path)
    im = Image.open(file_name)
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(new,image = tkimage)
    myvar.place(x=50, y=300)
    myvar.image = tkimage
    var1 = StringVar()
    var1.set("Input Image")
    font1 = TkFont.Font(family="verdana",size=15,weight="bold")
    label_inImage = Label( new, textvariable=var1,font= font1,bg="white",fg="blue")
    label_inImage.place(x=100, y=250)
   
   def process():
    global test_image_path
    var_out = StringVar()
    var_out.set("Result")
    font1_out = TkFont.Font(family="verdana",size=15,weight="bold")
    label_out = Label( new, textvariable=var_out,font= font1_out,bg="white",fg="blue")
    label_out.place(x=600, y=250)
    
    predicted_class = classify_logo(test_image_path)
    print("Predicted class:", predicted_class)
    var_result = StringVar()
    var_result.set(predicted_class)
    font1_result = TkFont.Font(family="verdana",size=25,weight="bold")
    label_result = Label( new, textvariable=var_result,font= font1_result,bg="white",fg="red")
    label_result.place(x=600, y=300)

   def classify_logo(image_path):
    print(image_path)
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

    _, predicted = torch.max(outputs.data, 1) #gives dimension 1 
    class_index = predicted.item() # not probability but returns 0 or 1
    classes = ['Genuine', 'fake']
    class_label = classes[class_index]

    return class_label
    


    var_result = StringVar()
    var_result.set(predicted_class)
    font1_result = TkFont.Font(family="verdana",size=25,weight="bold")
    label_result = Label( new, textvariable=var_result,font= font1_result,bg="white",fg="red")
    label_result.place(x=600, y=300)
   new= Toplevel(root) #opens window in root, stays in top level
   new.geometry("1000x750")
   new.title("New Window")
   #Create a Label in New window
   var = StringVar()
   font3 = TkFont.Font(family="verdana",size=30,weight="bold")
   label_new = Label( new, textvariable=var,font= font3,bg="white",fg="blue")
   var.set("FAKE LOGO ANALYZER")
   label_new.place(x=300, y=30)
   # button
   action2 = Button(new,text="Browse File", height= 3, width=15, fg='red', bg='cyan',command=clickMe)
   action2.place(x=50, y=130)
   # action.configure(state='disabled')

   # button
   action1 = Button(new,text="Detect", height= 3, width=15, fg='red', bg='cyan', command=process)
   action1.place(x=400, y=130)
   # action.configure(state='disabled')


 
 
def login():
    global flag
    username = username_entry.get()
    password = password_entry.get()

    if username == "user" and password == "abc":
        login_status.config(text="Login successful", fg="green")
        flag=1
        print(flag)
        open_win()
    else:
        login_status.config(text="Login failed. Try again.", fg="red")
        flag=0
        print(flag)
        
font1 = TkFont.Font(family="verdana",size=15,weight="bold")       
# Create labels and Entry widgets for username and password
username_label = Label(root, text="Username:",font= font1,fg="blue")
username_label.place(x=60, y=100)
password_label = Label(root, text="Password:",font= font1,fg="blue")
password_label.place(x=60, y=180)
username_entry = Entry(root)
username_entry.place(x=200, y=100)
password_entry = Entry(root, show="*****")  # Show '*' for password entry
password_entry.place(x=200, y=180)
# Create a Login button
login_button = Button(root, text="Login", height= 2, width=15, command=login)
login_button.place(x=180, y=260)
# Create a label to display login status
login_status = Label(root, text="", fg="black")




root.mainloop()
