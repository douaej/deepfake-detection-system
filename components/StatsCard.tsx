interface StatItem {
  label: string
  value: string | number
  unit?: string
  highlight?: boolean
}

interface StatsCardProps {
  stats: StatItem[]
}

export default function StatsCard({ stats }: StatsCardProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {stats.map((stat, idx) => (
        <div
          key={idx}
          className={`bg-card border rounded-lg p-3 text-center transition-all ${
            stat.highlight
              ? 'border-accent/50 bg-accent/5'
              : 'border-border hover:border-accent/30'
          }`}
        >
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
            {stat.label}
          </p>
          <p
            className={`text-xl font-bold mt-1 ${
              stat.highlight ? 'text-accent' : 'text-foreground'
            }`}
          >
            {stat.value}
            {stat.unit && <span className="text-xs ml-1">{stat.unit}</span>}
          </p>
        </div>
      ))}
    </div>
  )
}
