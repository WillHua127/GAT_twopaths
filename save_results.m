function save_results(filename, InitialX, InitialObjective, XTrace, ObjectiveTrace)
if length(InitialObjective) == size(InitialX, 1)
    XTrace = [InitialX; XTrace]; ObjectiveTrace = [InitialObjective; ObjectiveTrace];
end
T = XTrace; T.Objective = ObjectiveTrace;
[~, ia, ~] = unique(T, 'rows');
XTrace = XTrace(ia, :); ObjectiveTrace = ObjectiveTrace(ia);
InitialX = XTrace; InitialObjective = ObjectiveTrace;
save(filename, 'InitialX', 'InitialObjective');
end

