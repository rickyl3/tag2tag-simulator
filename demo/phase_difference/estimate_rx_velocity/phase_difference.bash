LOGFILE="$(dirname "$0")/../../../logs/$(ls ../../../logs | grep '\.json$' | tail -n 1)"

jq -r -s '
map(select(.tag != null and .v_n_spd != null and .v_n_dir != null and .v_n_spd != 0)) |
sort_by(.tag) |
group_by(.tag) |
map({
	tag: .[0].tag,
	v_n_spd: .[0].v_n_spd,
	v_n_dir: .[0].v_n_dir
}) |
map({
	tag: .tag,
	v_n_spd: .v_n_spd,
	v_n_dir: .v_n_dir
}) |
(["tag", "v_n_spd", "v_n_dir"]),  # header
(.[] | [.tag, .v_n_spd, .v_n_dir]) # rows
| @csv
' $LOGFILE > processed.csv
