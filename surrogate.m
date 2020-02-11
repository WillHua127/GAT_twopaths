function y = surrogate(HPT, network, target_cluster, singularity, dataset, public, implementation)
%% parse
suffix = ['--implementation', ' ', implementation, ' ', '--network', ' ', network, ' ', '--target_cluster', ' ', target_cluster, ' ', '--docker', ' ', num2str(singularity), ' '];
suffix = [suffix, '--dataset', ' ', dataset, ' '];
suffix = [suffix, '--public', ' ', num2str(public), ' '];
hp_names = HPT.Properties.VariableNames;
for i = 1: numel(hp_names)
    if strcmp(hp_names{i}, 'hidden')
        HPT{1, hp_names{i}} = 10 * HPT{1, hp_names{i}};
        hp_names{i} = 'hidden';
    end
end
HPT.Properties.VariableNames = hp_names;
for i = 1: numel(hp_names)
    thing = HPT{1, hp_names{i}};
    if isa(thing, 'cell')
        thing = thing{:};
    elseif isa(thing, 'categorical')
        thing = cellstr(thing);
        thing = thing{:};
    end
    suffix = [suffix, '--', hp_names{i}, ' ', num2str(thing), ' '];
end
% if validate
%     suffix = [suffix, '--optimizer RMSprop --validation 1'];
% else
%     suffix = [suffix, '--optimizer Adam --validation 0'];
% end
%% generate script
command = sprintf("python script_generator.py %s", suffix);
[status, cmdout] = system(command);
if status
	error(cmdout);
else
    script_name = cmdout(1: end - 1);
end
%% check status
identifier = regexp(script_name, '\d*', 'Match');
identifier = identifier{:};
completed = false; exception = false;
while ~completed && ~exception
    completed = checkin([identifier, '.txt'], 'completed');
    exception = checkin(identifier, 'exception');
    fprintf('result not found\n');
    pause(15);
end
%% return values
if exception
    try
        str_exception = load(['exception', '/', identifier, '.txt']);
		fprintf('%s', str_exception);
    catch ME
        fprintf('');
    end
    y = NaN;
else
    try
        y = 1 - load(['completed', '/', identifier, '.txt']);
        delete(['completed', '/', identifier, '.txt']);
    catch ME
        y = NaN;
    end
end
end

function flag = checkin(script_name, folder)
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(strfind(filenames{i}, script_name))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];
flag = ~isempty(filenames);
end
