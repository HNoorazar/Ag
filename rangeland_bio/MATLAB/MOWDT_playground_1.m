%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
pkg load signal
pkg load ltfat

research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = strcat(research_db, "/RangeLand_bio/");
rangeland_bio_data = strcat(rangeland_bio_base, "Data/");
min_bio_dir = strcat(rangeland_bio_data, "Min_Data/");

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = strcat(rangeland_base,"reOrganized/");

bio_reOrganized = strcat(rangeland_bio_data, "reOrganized/");
%  os.makedirs(bio_reOrganized, exist_ok=True);

bio_plots = strcat(rangeland_bio_base, "plots/");
%  os.makedirs(bio_plots, exist_ok=True);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
%
%  read anpp file
%
filename = strcat(bio_reOrganized, "bpszone_ANPP_no2012.csv");
anpp = readmatrix(filename); %  did not work in Octav
anpp = csvread(filename);
disp(anpp(1:5, :));

anpp = csv2cell(filename);
disp(anpp(1:5, :));



fids = cell2mat(anpp(2:end, 2)); 
disp(fids(1:5));
a_fid_anpp = anpp(fids == 1, :);

%  pkg install -forge statistics
%  pkg install -forge wecg


wname = 'db4';
% Number of decomposition levels
J = 6;
w = modwt(a_fid_anpp, wname, J);