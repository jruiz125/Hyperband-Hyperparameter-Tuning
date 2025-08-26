function [cont,cXposX,cXposY] = intersegments( cXpos)
%calcular el numero de segmentos de las intersecciones

lcXpos=length(cXpos(1,:));

if lcXpos>cXpos(2,1)+1
    %M?s de 1 segmento
    cont=1;%No. de segmentos
    lengthsegment=cXpos(2,1)+1;
    cXposX=cXpos(1,2:lengthsegment);
    cXposY=cXpos(2,2:lengthsegment);
    cXposX(end+1)=NaN;
    cXposY(end+1)=NaN;
    lastcol=lengthsegment;
    
    while lcXpos>lastcol
        cont=cont+1;
        
        lec1=lastcol+2;
        lec2=lastcol+cXpos(2,lastcol+1)+1;
        
        lim1=lec1-1;
        lim2=lec2-1;
        cXposX(lim1:lim2)=cXpos(1,lec1:lec2);
        cXposY(lim1:lim2)=cXpos(2,lec1:lec2);
        cXposX(end+1)=NaN;
        cXposY(end+1)=NaN;
        
        lastcol=lec2;
    end;
    
else
    %1 segmento
    cont=1;%No. de segmentos
    cXposX=cXpos(1,2:end);
    cXposY=cXpos(2,2:end);
end;


end

