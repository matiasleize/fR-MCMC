## oregonator differential equation



function xdot = f(x, z)

  C_1 = 1;
  C_2 = 1/19;  
  xdot = zeros (5,1);
 
  #x = x(1);
  %y = x(2);
  %v = x(3);
  #w = x(4);
  #r = x(5);
  
  gama= @(r,C_1,C_2) ((1+C_2.*r) * ((1+C_2.*r).^2 - C_1)) / (2 .* C_1 .* C_2 .* r);


  xdot(1) = (-x(4) + x(1)**2 + (1 + x(3))*x(1) - 2*x(3) + 4*x(2))/(1+z);
  
  xdot(2) = -(x(3)*x(1)*gama(x(5),C_1,C_2) - x(1)*x(2)  + 4*x(2) - 2*x(2)*x(3))/(1+z);
  
  xdot(3) = -(x(3) * (x(1)*gama(x(5),C_1,C_2)  + 4 - 2*x(3)))/(1+z);
  
  xdot(4) = (x(4) * (-1 + x(1) + 2*x(3)))/(1+z);
  
  xdot(5) = (-x(5) * gama(x(5),C_1,C_2) * x(1))/(1+z);

endfunction
