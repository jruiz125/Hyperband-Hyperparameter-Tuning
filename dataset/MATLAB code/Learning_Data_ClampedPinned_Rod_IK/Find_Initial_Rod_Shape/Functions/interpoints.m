function [npoints,intX,intY] = interpoints(lin1X,lin1Y,lin2X,lin2Y)
%calcular el numero de puntos de interseccion de 2 lineas y su posicion
%por barrido
npoints=0;
intX=NaN;
intY=NaN;

for ii=2:length(lin1X)
        Ax=lin1X(ii-1);
        Ay=lin1Y(ii-1);
        Bx=lin1X(ii);
        By=lin1Y(ii);
        m1=(By-Ay)/(Bx-Ax);
        n1=Ay-m1*Ax;
        
    for jj=2:length(lin2X)
        Cx=lin2X(jj-1);
        Cy=lin2Y(jj-1);
        Dx=lin2X(jj);
        Dy=lin2Y(jj);
        m2=(Dy-Cy)/(Dx-Cx);
        n2=Cy-m2*Cx;
        
        %coordenadas de la intersección
        xi=(n1-n2)/(m2-m1);
        yi=m1*xi+n1;
        
        %         figure(8);
        %         cla;
        %         plot(lin1X,lin1Y);hold on;
        %         plot(lin2X,lin2Y);
        %         plot([Ax Bx],[Ay By],'k', 'LineWidth',3);
        %         plot([Cx Dx],[Cy Dy],'k', 'LineWidth',3)
        %         plot(xi,yi,'o')
        %         pause(0.001)
        
        %validez de la intersección
        vAB=([Bx-Ax By-Ay]);
        vAI=([xi-Ax yi-Ay]);
        vCD=([Dx-Cx Dy-Cy]);
        vCI=([xi-Cx yi-Cy]);
        
        if vAI==vAB
            npoints=npoints+1;
            intX(npoints)=Bx;
            intY(npoints)=By;
        elseif vAI==[0 0]
            npoints=npoints+1;
            intX(npoints)=Ax;
            intY(npoints)=Ay;
        elseif vCI==vCD
            npoints=npoints+1;
            intX(npoints)=Dx;
            intY(npoints)=Dy;
        elseif vCI==[0 0]
            npoints=npoints+1;
            intX(npoints)=Cx;
            intY(npoints)=Cy;
        elseif (vAB/vAI)>1&&(vCD/vCI)>1
            npoints=npoints+1;
            intX(npoints)=xi;
            intY(npoints)=yi;
        end;
    end;
end;
