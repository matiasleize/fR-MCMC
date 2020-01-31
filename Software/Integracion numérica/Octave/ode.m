%% gama se cambia directo de f.m

z = linspace (0, 6, 10000);

x_00 = -0.339;
y_00 = 1.246;
v_00 = 1.64;
w_00 = 1 + x_00 + y_00 - v_00; 

x0 = [x_00; y_00; v_00; w_00];

y = lsode ("f", x0, z); 

%% Funcíon auxiliar al integrar v para el calculo de H
FF = cumtrapz(z, y(:,3)'./(z+1)); 

## Calculamos la constante de Hubble en función del redshift.
H = (z+1).^2 .* e.^(-FF);
%plot(z,H)

## Calculamos la distancia luminosa

H_0 =  74.2
C =  299792458.0

d_c = (C / H_0) * cumtrapz(z,H.**(-1));
d_L = d_c .* (1+z);

%plot(z,d_c)

%% Guardamos los datos para importarlos en Python
B = zeros(length(H),3);
B(:,1)=z;
B(:,2)=H;
B(:,3)=d_L;

dlmwrite('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/Datos_octave/datos_octave.txt',B,'delimiter', '\t');


#%% Ploteo de la constante

cte = y(:,3) .+ y(:,4) .-(y(:,1) .+ y(:,2));
plot(z,cte)
