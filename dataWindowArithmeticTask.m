% Dividing clean data - Rest 

n=10*128 ; % 10s 
p=0; % 0s overlap 

%% store all the files in the directory 
matFiles = dir('*.mat'); 
numfiles = length(matFiles);


for k = 1:numfiles  % Loop through all the files 
  load(matFiles(k).name); 
  X=clean_data;   % store all the channels in the clean data 
  for i = 1:14   % loop through different channels 
      data = buffer(X(i,:),n,p); % selecting single channel and dividing it into 10s windows 
      newData = "newdata";
      S.(newData) = data;
      
      fileName = "/home/muditdhawan/Desktop/EEG/CleanData_TDC/Music/features/dataWindowChannel" + i +".mat";
      if k==1
          save(fileName, '-struct', 'S') 
      else 
          temp = load(fileName).(newData)  ;
          temp = [temp data];
          R.(newData) = temp;
          save(fileName, '-struct', 'R');
      end
      
  end 

end

