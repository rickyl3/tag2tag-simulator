LOGFILE="$(dirname "$0")/../../logs/$(ls ../../logs | grep '\.json$' | tail -n 1)"

jq -r -s '
map(select(.tag != null and .phase_ang != null and .phase_diff != null)) |
sort_by(.tag) |
group_by(.tag) |
map({
	tag: .[0].tag,
	phase_ang: .[0].phase_ang,
	phase_diff: .[0].phase_diff,
}) |
map({
	tag: .tag,
	phase_ang: .phase_ang,
	phase_diff: .phase_diff
}) |
(["tag", "phase_ang", "phase_diff"]),  # header
(.[] | [.tag, .phase_ang, .phase_diff]) # rows
| @csv
' $LOGFILE > processed.csv
