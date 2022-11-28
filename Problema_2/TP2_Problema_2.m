% TP2 - Problema 2
close all, clear, clc

%%
f = imread('monedas.jpg');
figure, imshow(f, []);

%%
fgray = rgb2gray(f);
figure, imshow(fgray, []);

%%
fgrayadj = imadjust(fgray);
figure, imshow(fgrayadj, []);

%%
fcanny = edge(fgrayadj, 'Canny', 0.1, 5);
figure, imshow(fcanny, []);

%%
fdilate = imdilate(fcanny, strel('disk', 5));
figure, imshow(fdilate, []);

%%
ffill = imfill(fdilate, 'holes');
figure, imshow(ffill, []);

%%
fopen = imopen(ffill, strel('disk', 20));
figure, imshow(fopen, []);

%%
[L, num] = bwlabel(fopen);
flabel = label2rgb(L, jet(num), [0.5,0.5,0.5]);
figure(), imshow(flabel, []);

%%
props = regionprops(L, 'Area', 'Perimeter');
propsArea = [props.Area];
propsPer = [props.Perimeter];
rho = 4 .* pi .* (propsArea ./ propsPer.^2);

Laux = L;
for i = 1:num
   if(rho(i) < 0.9)
       Laux(Laux == i) = 0;
   end
end
figure, imshow(imbinarize(Laux), []);

%%
[Lmon, numMon] = bwlabel(Laux);
propsMonedas = regionprops(L, 'Area', 'Perimeter', 'EquivDiameter');
monedasArea = [propsMonedas.Area];
monedasPerimeter = [propsMonedas.Perimeter];
monedasEquivDiameter = [propsMonedas.EquivDiameter];

ind = kmeans(monedasEquivDiameter', 3);
Lfinal = Lmon;
Lfinal(ismember(Lmon, find(ind == 1))) = 1;   
Lfinal(ismember(Lmon, find(ind == 2))) = 2; 
Lfinal(ismember(Lmon, find(ind == 3))) = 3; 

fclas = label2rgb(Lfinal, jet(3), [0.5,0.5,0.5]);
figure(), imshow(fclas,[])

