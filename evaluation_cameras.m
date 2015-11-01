data = csvread("Results/resultList_cameras_raw_data.csv");
result = zeros(rows(data)/10, 2*columns(data));

k = 1;
for i = 1:10:rows(data) 
	for j = 1:columns(data)
		col = data([i:i+9],j);

		if(!isnumeric(col(1,1)) || col(1,1) == -1) 
			result(k, 2*j-1) = -1;
			result(k, 2*j) = -1;
		else
			result(k, 2*j-1) = mean(col);
			result(k, 2*j) = std(col);
		endif
	endfor
	k++;	
endfor

csvwrite("evaluation_cameras.csv", result);

evaluation_small = zeros(80, 2*columns(data));

k=1;
for t = 0.05:0.05:4
	xscale(k) = t;
	for j = 1:columns(data)
		for i = 1:rows(data) 
			if(isnumeric(data(i,j)) && data(i,j) > 0 && data(i,j) < t)
				evaluation_small(k, 2*j -1) += data(i,j);
				evaluation_small(k, 2*j)++;
			endif	
		endfor 
		if(evaluation_small(k, 2*j) > 0) 
			evaluation_small(k, 2*j - 1) /= evaluation_small(k, 2*j);
		endif
	endfor
	k++;
endfor

#size(evaluation_small)
#size(xscale)

#evaluation_small



msize=8;
lwidth=1.5;
fontsize=20;

h1=figure("Position",[0,0,800,600]);
#plot (xaxis_t, data(:,1), ":ko", "markersize", msize, "linewidth", lwidth, xaxis_t, data(:,3), ":go", "markersize", msize, "linewidth", lwidth, xaxis_t, data(:,5), ":bo", "markersize", msize, "linewidth", lwidth, xaxis_t, data(:,7), ":ro", "markersize", msize, "linewidth", lwidth);
plot (xscale, evaluation_small(:,1), ":ko", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,5), ":go", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,9), ":bo", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,13), ":ro", "markersize", msize, "linewidth", lwidth);

title ("Mean translation distances up to a threshold");
xlabel ("Threshold", "fontsize", fontsize);
ylabel ("Mean translation distance", "fontsize", fontsize);
l = legend ("final result", "from points", "from line homog.", "from point homog.", "location", "northwest");
set (l, "fontsize", fontsize);
set(gca, "fontsize", fontsize)

h2=figure("Position",[0,0,800,600]);
plot (xscale, evaluation_small(:,3), ":ko", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,7), ":go", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,11), ":bo", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,15), ":ro", "markersize", msize, "linewidth", lwidth);

title ("Mean rotation distances up to a threshold");
xlabel ("Threshold", "fontsize", fontsize);
ylabel ("Mean rotation distance", "fontsize", fontsize);
l = legend ("final result", "from points", "from line homog.", "from point homog.", "location", "southeast");
set (l, "fontsize", fontsize);
set(gca, "fontsize", fontsize)

h3=figure("Position",[0,0,800,600]);
plot (xscale, evaluation_small(:,2), ":ko", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,6), ":go", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,10), ":bo", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,14), ":ro", "markersize", msize, "linewidth", lwidth);

title ("Number of estimations with translation distances below a threshold");
xlabel ("Threshold", "fontsize", fontsize);
ylabel ("Number of estimations", "fontsize", fontsize);
l = legend ("final result", "from points", "from line homog.", "from point homog.", "location", "southeast");
set (l, "fontsize", fontsize);
set(gca, "fontsize", fontsize)

h4=figure("Position",[0,0,800,600]);
plot (xscale, evaluation_small(:,4), ":ko", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,8), ":go", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,12), ":bo", "markersize", msize, "linewidth", lwidth, xscale, evaluation_small(:,16), ":ro", "markersize", msize, "linewidth", lwidth);

title ("Number of estimations with rotation distances below a threshold");
xlabel ("Threshold", "fontsize", fontsize);
ylabel ("Number of estimations", "fontsize", fontsize);
l = legend ("final result", "from points", "from line homog.", "from point homog.", "location", "southeast");
set (l, "fontsize", fontsize);
set(gca, "fontsize", fontsize)
