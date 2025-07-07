import { LineChart, Line, XAxis, YAxis } from 'recharts';

const data = [
  { name: 'A', value: 30 },
  { name: 'B', value: 20 },
];

export default function MetricsChart() {
  return (
    <LineChart width={300} height={200} data={data}>
      <Line type="monotone" dataKey="value" stroke="#8884d8" />
      <XAxis dataKey="name" />
      <YAxis />
    </LineChart>
  );
}
