LOGFILE="$(dirname "$0")/../../logs/$(ls ../../logs | grep '\.json$' | tail -n 1)"

jq -r -s '
map(select(.tag != null and .phase_ang != null and .phase_diff != null and .doppler_freq != null)) |
sort_by(.tag) |
group_by(.tag) |
map({
	tag: .[0].tag,
	phase_ang: .[0].phase_ang,
	phase_diff: .[0].phase_diff,
	doppler_freq: .[0].doppler_freq
}) |
map({
	tag: .tag,
	phase_ang: .phase_ang,
	phase_diff: .phase_diff,
	doppler_freq: .doppler_freq
}) |
(["tag", "phase_ang", "phase_diff", "doppler_freq"]),  # header
(.[] | [.tag, .phase_ang, .phase_diff, .doppler_freq]) # rows
| @csv
' $LOGFILE > processed.csv
