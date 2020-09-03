%% gama se cambia directo de f.m

z = linspace (0, 3, 100000);

x_00 = -0.339;
y_00 = 1.246;
v_00 = 1.64;
w_00 = 1 + x_00 + y_00 - v_00; 
r_00 = 41;

x0 = [x_00; y_00; v_00; w_00; r_00];

y = lsode ("f", x0, z); 

plot(z,y(:,2))

%% Funcíon auxiliar al integrar v para el calculo de H
FF = cumtrapz(z, y(:,3)'./(z+1)); 

## Calculamos la constante de Hubble en función del redshift.
E = (z+1).^2 .* e.^(-FF);
plot(z,E)

## Calculamos la distancia luminosa

H_0 =  73.48
C =  299792458.0

d_c = (C / H_0) * cumtrapz(z,E.**(-1));
d_L = d_c .* (1+z);

%plot(z,d_c)

%% Guardamos los datos para importarlos en Python
B = zeros(length(E),2);
B(:,1)=z;
B(:,2)=E;
%B(:,3)=d_L;

#dlmwrite('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/analisis_int_solve_ivp/datos_octave.txt',B,'delimiter', '\t');
dlmwrite('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Octave/datos_octave.txt',B,'delimiter', '\t');

#%% Ploteo de la constante

cte = y(:,3) .+ y(:,4) .-(y(:,1) .+ y(:,2));
plot(z,cte)
