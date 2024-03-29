% create test MATLAB data

%% File 1 - flat data
col_nums = [1; 2; 3; 4; 5];
row_nums = [1 2.2 3];
mat_nums = [1 2 3; 4 5 6];

save('test_numbers.mat', '-v7.3', 'col_nums', 'row_nums', 'mat_nums');

%% File 2 - structure
x   = [];
x.r = row_nums;
x.c = col_nums;
x.m = mat_nums;

save('test_struct.mat', '-v7.3', 'x');

%% File 3 - Enumerators
enum = [Gender.male, Gender.female, Gender.female, Gender.circ_male, Gender.circ_male, ...
    Gender.circ_male]; % [2 1 1 4 4 4]

save('test_enums.mat', '-v7.3', 'enum');

%% File 4 - Cell Array
cdat = {row_nums, col_nums, mat_nums, 'text', 'longer text', '', []};

save('test_cell_array.mat', '-v7.3', 'cdat');

%% File 5 - nested data
data     = [];
data.x   = x;
data.y   = x;
data.y.r = data.y.r + 10;
data.y.c = data.y.c + 20;
data.y.m = data.y.m + 30;
data.z   = [];
data.z.a = [1 2 3];
data.z.b = enum;
data.c   = cdat;
data.nc  = {row_nums, col_nums, x};

save('test_nested.mat', '-v7.3', 'col_nums', 'row_nums', 'mat_nums', 'x', 'enum', 'cdat', 'data');

%% File 6 - raw binary data
% big endian
fid = fopen('test_big_endian.bin', 'wb', 'ieee-be');
assert(fid ~= -1);
fwrite(fid, [uint32(3), uint32(2^32 - 3)], 'uint32');
fwrite(fid, [uint32(3), uint32(2^16 + 3)], 'int32');
fwrite(fid, single([0 1.5 -2.333333333333333333 pi]), 'single');
fwrite(fid, [0 -1.5 pi exp(1)], 'double');
fclose(fid);
% little endian
fid = fopen('test_little_endian.bin', 'wb', 'ieee-le');
assert(fid ~= -1);
fwrite(fid, [uint32(3), uint32(2^32 - 3)], 'uint32');
fwrite(fid, [uint32(3), uint32(2^16 + 3)], 'int32');
fwrite(fid, single([0 1.5 -2.333333333333333333 pi]), 'single');
fwrite(fid, [0 -1.5 pi exp(1)], 'double');
fclose(fid);