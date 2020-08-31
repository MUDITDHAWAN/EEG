%% Loading clean data 
matFiles = dir('*.mat'); 
numfiles = length(matFiles);

%% Parse through all the data files 
for j = 1:numfiles 
    X = load(matFiles(j).name).("newdata"); 
    s = size(X,2);
    %% Loop through all the channels 
    for i = 1: s 
        featureVector = [] ;
        Fs = 128 ;
        t = 1: s; %% creating time axis
        v = X(:,i);
        v = v';
        vdash = diff(v) ;
        vdoubledash = diff(vdash);

        %% ploting one channel
        % figure(1);
        % plot(t,X(1,:));
        % plot(t,vdash);
        %%  Statistical Features 

        % Standard Deviation
        stdSignal = std(v);

        featureVector = [ featureVector ; stdSignal ];

        % LAR: Latency to Amplitude 
        [smax, tsmax] = max(v);
        lar = tsmax / smax ;

        featureVector = [ featureVector ; lar ];

        % First Difference 
        fdiff = sum(vdash);

        featureVector = [ featureVector ; fdiff] ;

        % Normalized Fist Difference 
        nfdiff = fdiff / stdSignal ;

        featureVector = [ featureVector ; nfdiff ] ;

        % Pk-Pk Signal Value 
        [smin, tsmin ] = min(v);

        spp = smax - smin ;

        featureVector = [ featureVector ; spp ] ;

        % Pk-Pk Time Window 
        tpp = tsmax + tsmin ;

        featureVector = [ featureVector ; tpp ] ;

        % Signal Power 

        rmsSignal = rms(v) ;
        featureVector = [ featureVector ; rmsSignal ]; 

        pRMS = rmsSignal^2 ;

        powbp = bandpower(v,Fs,[0 Fs/2]) ;
        featureVector = [ featureVector ; powbp];

        % Mean Value 
        signalMean = mean(v);
        featureVector = [ featureVector ; signalMean ];

        % Kurtosis 
        k = kurtosis(v) ;

        featureVector = [ featureVector ; k];
        %%% Hjorth Features 
        % activity = std ^2 

        % Mobility 
        varSignal = var(v) ;
        varDashSignal = var(vdash) ;

        mobility = varDashSignal / varSignal ;

        featureVector = [ featureVector ; mobility ] ;
        
        % Complexity 
        varDoubleDashSignal = var(vdoubledash); 
        mobilityvdash = var(vdoubledash) / var(vdash);

        complexity = mobilityvdash / mobility ;
        
        featureVector = [ featureVector ; complexity ];

        %%% Non Stationary Index 
        normsig = v/std(v);
        
        windowfornsi = buffer(normsig,128*1.5,0)';
        meannsi = mean(windowfornsi) ;
        nsi = std(meannsi) ;
        featureVector = [ featureVector ; nsi ];
        
        % Fractional Dimension - boxcount method 
        % [n, r] = boxcount(v,'plot');
        
%         %% Higher Order Spectra (HOS)
%         [Bspec,waxis] = bispecd(v,  128, 5, 8, 50); % bispectrum 
%         figure(1);
%         surf(waxis, waxis , abs(Bspec)); % 3D magnotude curve
%         [Bcoher,waxis] = bicoher(v,  128, 5, 8, 50) ; % bicoherence 
%         figure(2);
%         surf(waxis, waxis , abs(Bcoher)); % 3D magnotude curve
        
        %% Higher Order Crossings (HOC)
        ch=v ; k = 9; t = 1280 ;
            condn = zeros(k,t);
            condn(1,:) = ch;

            for ind=2:k
               condn(ind,ind:t) = diff(ch,ind-1) ;
            end
            X_mat=zeros(k,t);
            for ind=1:k
               for ind2=ind:t
                   if condn(ind,ind2)>=0
                       X_mat(ind,ind2) = 1;
                   else
                       X_mat(ind,ind2) = 0;
                   end
               end
            end

            Dk = zeros(1,9);
            for ind=1:k
                rsum = 0;
                for ind2=2:t
                    rsum = rsum+ (X_mat(ind,ind2)-X_mat(ind,ind2-1)).^2;
                end
                Dk(ind)=rsum;
            end
        
        hocFeature = Dk;
        
        featureVector = [ featureVector ;hocFeature'] ;
        
        %% Frequency domain Features 
        
        [pxx,f] = pwelch(v,128,0,[],Fs);
        
        % calculating avg psd over f2- f1 = 2 Hz
        hz2band = buffer(pxx,4,0)';
        avghz2band = mean(hz2band) ;
        
        avgpower = mean(avghz2band);
        featureVector = [ featureVector ;avgpower ] ;
        
        minpower= min(avghz2band);
        featureVector = [ featureVector ; minpower] ;
        
        maxavgpower = max(avghz2band);
        featureVector = [ featureVector ;maxavgpower ];
        
        varavgpower = var(avghz2band);
        featureVector = [ featureVector ; varavgpower ];
        
        % Band Power 
        bpower = bandpower(pxx,f,'psd');
        featureVector = [ featureVector ; bpower ];
        
        % power features from different frequency bands
        
        deltapower = mean(pxx(2*1+1:2*4+1)) ;
        thetapower = mean(pxx(2*4+1:2*18+1)) ;
        slowalphapower = mean(pxx(2*8+1:2*10+1)) ;
        alphapower = mean(pxx(2*8+1:2*12+1)) ;
        betapower = mean(pxx(2*12+1:2*30+1)) ;
        gammapower = mean(pxx(2*20+1:2*64+1)) ;
        
        ratiomeanalphabybeta = mean(alphapower)/ mean(betapower);
        
        featureVector = [ featureVector ; ratiomeanalphabybeta ];
            
        
        %% Hilbert-Huang Spectrum (HHS)
%         [imf, residual] = emd(v,'Display',0);
% 
%         [hs, fhs, thst ] = hht(imf,Fs,'FrequencyLimits',[0 20]);
%         
%         ampHS = abs(hs);
%         phaseHS = angle(hs);
        
        %% Discrete Wavelt Transform -  2D thnk of it as image - use different channels and time axis 

        % 1D 
        wname = 'db4';
        % [cA,cD] = dwt(v,wname); 

        n = 4;  % taking D4 coeff
        [c,l] = wavedec(v,n,wname) ;

        approx = appcoef(c,l,wname);

        [cd1,cd2,cd3,cd4] = detcoef(c,l,[1 2 3 4]);

        %entropy shannon
        entropyWaveletShannon = wentropy(cd4,'shannon') ;
        featureVector = [ featureVector ; entropyWaveletShannon ] ;

        % entropy log
        entropyWaveletLog = wentropy(cd4,'log energy');
        featureVector = [ featureVector ;  entropyWaveletLog ] ;
        % energy 
        energySigWavelet = sqrt(sum(abs(cd4).^2,2));
        featureVector = [ featureVector ; energySigWavelet ] ;
        % rms
        rmsWavelet = rms(cd4); 
        featureVector = [ featureVector ; rmsWavelet ] ;
        
        newData = "newdata";
      S.(newData) = featureVector ;
      
      fileName = "/home/muditdhawan/Desktop/EEG/CleanData_TDC/Rest/finalData/featureChannel"+j+".mat";
      if i==1
          save(fileName, '-struct', 'S') 
      else 
          temp = load(fileName).(newData)  ;
          temp = [temp featureVector];
          R.(newData) = temp;
          save(fileName, '-struct', 'R');
      end
      
%     break;
    end 
% break;
end

