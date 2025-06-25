%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
%%% I know the title says adaptive levels (how many levels to decompose)
%%% but since length of our signals are identical, those levels will be identical

// pkg load signal % this is for Octav
// pkg load ltfat

research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = strcat(research_db, "/RangeLand_bio/");
rangeland_bio_data = strcat(rangeland_bio_base, "Data/");
min_bio_dir = strcat(rangeland_bio_data, "Min_Data/");

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = strcat(rangeland_base,"reOrganized/");

bio_reOrganized = strcat(rangeland_bio_data, "reOrganized/");
% os.makedirs(bio_reOrganized, exist_ok=True);

bio_plots = strcat(rangeland_bio_base, "plots/");
% os.makedirs(bio_plots, exist_ok=True);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
%
%  read anpp file
%
filename = strcat(bio_reOrganized, "bpszone_ANPP_no2012.csv");
% anpp = readmatrix(filename); %  did not work in Octav
% anpp2 = csvread(filename, R1=1); 

% works great. shows variable names and reads string columns
anpp_table = readtable(filename,'PreserveVariableNames', true);
head(anpp_table, 2)
%% disp(anpp_table(1:5, :));

fids = (anpp_table.fid);
unique_fids = unique(fids);

a_fid_anpp_table = anpp_table(anpp_table.fid == 1, :);
size(a_fid_anpp_table)


wname = 'db4';
y_var_ = "mean_lb_per_acr";

signal_ = a_fid_anpp_table.(y_var_);
% Number of decomposition levels
% This will be 2 by vector of size 40.
level_num = wmaxlev(length(signal_), wname);
level_num = 5;
w2 = modwt(signal_, wname, 2);
w5 = modwt(signal_, wname, 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%
%%%%%%%%%     Compute variances at each scale
%%%%%%%%%
% var(w2, flag=0, dim=2); 
% flag means the denimonitor to be N or N-1
% dim means along which axis.
variabces2 = var(w2, 0, 2);
variabces5 = var(w5, 0, 2);

%%%%%%%%%
%%%%%%%%%     track variance shifts
%%%%%%%%%
%%%%%%%%% of course the low_freq below wont work if level_num is less than 2.
high_freq_levels = 1:2;
low_freq_levels = 4:level_num;

high_var2 = sum(variabces2(1));
low_var2 = sum(variabces2(2));

high_var5 = sum(variabces5(1:2));
low_var5 = sum(variabces5(4:5));


shift_ratio5 = low_var5 / high_var5;


figure;
for j = 1:5
    subplot(5, 1, j);
    plot(w5(j, :));
    title(['Wavelet coefficients at level ' num2str(j)]);
    xlabel('time')
    ylabel('coefficient value')
end



figure(2);
for j = 1:2
    subplot(2, 1, j);
    plot(w2(j, :));
    title(['Wavelet coefficients at level ' num2str(j)]);
    xlabel('time')
    ylabel('coefficient value')
end